// THE CRYSTAL THINKS: a lookahead over the crystal's own reads, and the
// no-repetition rule. Both are the FPGA's, not the glass's — the glass only ever answers "which move, and how good is this board".
//
// THINKING, WHEN UNSURE: when the brightest legal move holds under `share` of the top three's light, the crystal looks ahead. Level 0 is
// our move (our top W0 legal cells by light), level 1 their reply on the board that leaves them (their top W1), level 2 ours again, and so
// on; past the last level the board is judged by the value readout (tr.value: win, draw and loss cells) from our side. Our levels take the
// best, theirs the worst for us; a jump chain in progress keeps the same mover at the same level. Every look is one more read of the
// crystal (four passes). Thinking when unsure at widths 3 × 1 × 1: 155 won / 35 drawn / 10 lost against the crystal alone, +319 Elo.
//
// NO REPETITION WHEN NOT LOSING: the positions since the last irreversible move (a capture or any man move) are kept; if the crystal's move
// would recreate one while its material is not behind, the brightest of its top 5 legal cells that does not repeat is played instead —
// if it has at least a quarter of the chosen move's light (a far dimmer move is the map saying it does not believe in it).
import type { CheckersMove, CheckersState, MoveReading } from "./board.js";
import { checkers as C } from "./board.js";

export interface ThinkOptions {
  /** "unsure": think only when the brightest move holds under `share` of the top three's light; "always"; "off" */
  mode: "unsure" | "always" | "off";
  share: number;
  /** widths per level: [3, 1, 1] is 3-ply (our top 3, their top reply, our top answer); [3, 3] 2-ply three wide */
  widths: number[];
  /** the crystal's move-map light for a board (one read: four passes) */
  look: (st: CheckersState) => Promise<Float64Array>;
  /** the value cells (win, draw, loss) */
  valueCells: number[][];
  /** false once the game has moved on: the search stops */
  alive?: () => boolean;
  /** NARROW THE SURE LINES (as chess's): inside the tree, a board whose brightest move holds at least this share of its top
   *  W[lv] light follows only that move — `ours` on the crystal's turns, `theirs` on yours; 1 never narrows; null: off */
  narrow?: { ours: number; theirs: number } | null;
  /** each board the look-ahead reads, as the line of moves that reaches it from the position thought about (the ghost board) */
  onLook?: (line: CheckersMove[]) => void;
  /** each board's value once read, as the crystal's expected score (0 lost … 1 won): the value meter */
  onValue?: (v: number) => void;
}

export interface ThinkNote {
  looked: number;
  widths: string;
  share: number;
  scores: { uci: string; v: number }[];
  wasUci?: string;
}

/** the value readout's cells from the crystal file's tr.value blocks ([row0, col0, row1, col1] on the 32 × 32 grid, inclusive) */
export function valueCells(model: Record<string, unknown>): number[][] | null {
  const v = ((model["tr"] as Record<string, unknown> | undefined)?.["value"] ?? null) as Record<string, { block?: number[] }> | null;
  if (!v) return null;
  const cells = (b?: number[]): number[] => {
    const out: number[] = [];
    if (!b || b.length < 4) return out;
    for (let r = b[0]!; r <= b[2]!; r++) for (let c = b[1]!; c <= b[3]!; c++) out.push(r * 32 + c);
    return out;
  };
  const g = [cells(v["win"]?.block), cells(v["draw"]?.block), cells(v["loss"]?.block)];
  return g.every((x) => x.length) ? g : null;
}

/** the expected score for the side to move, from the value cells: (win + draw / 2) over all three */
export function valueOf(p: ArrayLike<number>, g: number[][]): number {
  const V = g.map((cs) => cs.reduce((a, c) => a + Math.max(0, +(p[c] ?? 0)), 0)), S = Math.max(V[0]! + V[1]! + V[2]!, 1e-30);
  return (V[0]! + 0.5 * V[1]!) / S;
}

function topLegal(st: CheckersState, p: ArrayLike<number>): { cell: number; m: CheckersMove; w: number }[] {
  const seen = new Map<number, { cell: number; m: CheckersMove; w: number }>();
  for (const e of C.viewMoves(st)) if (!seen.has(e.cell)) seen.set(e.cell, { cell: e.cell, m: e.m, w: Math.max(0, +(p[e.cell] ?? 0)) });
  return [...seen.values()].sort((a, b) => b.w - a.w || a.cell - b.cell);
}

/** the lookahead: the reading it plays (its own, when sure or when the search agrees) and what it looked at */
export async function think(st: CheckersState, p: ArrayLike<number>, rd: MoveReading, o: ThinkOptions): Promise<{ rd: MoveReading; note: ThinkNote | null }> {
  if (o.mode === "off" || !rd.move) return { rd, note: null };
  const W = o.widths.length ? o.widths : [3, 1, 1], all = topLegal(st, p);
  if (all.length < 2) return { rd, note: null };
  const t3 = all.slice(0, 3), tot = t3.reduce((a, c) => a + c.w, 0), share = tot > 0 ? t3[0]!.w / tot : 1;
  if (o.mode === "unsure" && share >= o.share) return { rd, note: null };
  const cache = new Map<string, Float64Array>();
  const look = async (b: CheckersState): Promise<Float64Array> => {
    const k = C.str(b);
    let v = cache.get(k);
    if (!v) {
      if (o.alive && !o.alive()) throw new Error("cancelled");
      v = await o.look(b);
      cache.set(k, v);
    }
    return v;
  };
  const sv = async (b: CheckersState, us: boolean, lv: number, line: CheckersMove[]): Promise<number> => {
    if (!C.moves(b).length) return us ? 0 : 1; // the side to move has no move: it has lost
    o.onLook?.(line);
    const pb = await look(b);
    if (o.onValue && o.valueCells) {
      const e = valueOf(pb, o.valueCells);
      o.onValue(us ? e : 1 - e);
    }
    if (lv >= W.length) {
      const e = valueOf(pb, o.valueCells);
      return us ? e : 1 - e;
    }
    let best: number | null = null;
    let list = topLegal(b, pb).slice(0, W[lv]);
    const bar = o.narrow ? (us ? o.narrow.ours : o.narrow.theirs) : 1, sum = list.reduce((a, c) => a + c.w, 0);
    if (list.length > 1 && bar < 1 && sum > 0 && list[0]!.w / sum >= bar) list = list.slice(0, 1); // sure: only its brightest move
    for (const c of list) {
      const n = C.make(b, c.m), same = n.turn === b.turn, v = await sv(n, same ? us : !us, same ? lv : lv + 1, [...line, c.m]);
      best = best === null ? v : us ? Math.max(best, v) : Math.min(best, v);
    }
    return best ?? (us ? 0 : 1);
  };
  const sc: { c: { cell: number; m: CheckersMove; w: number }; v: number }[] = [];
  for (const c of all.slice(0, Math.max(1, W[0]!))) {
    const n = C.make(st, c.m), same = n.turn === st.turn;
    sc.push({ c, v: await sv(n, same, same ? 0 : 1, [c.m]) });
  }
  let best = sc[0]!;
  for (const x of sc) if (x.v > best.v + 1e-9) best = x;
  const note: ThinkNote = { looked: cache.size, widths: W.join("x"), share: +share.toFixed(3), scores: sc.map((x) => ({ uci: C.uci(x.c.m)!, v: +x.v.toFixed(3) })) };
  if (best.c.cell === rd.cell) return { rd, note };
  const n = C.make(st, best.c.m), cm = C.cellMove(best.c.cell);
  return { rd: { ...rd, cell: best.c.cell, move: best.c.m, uci: C.uci(best.c.m), state: n, from: cm.from, to: cm.to, score: best.c.w, clean: false }, note: { ...note, wasUci: rd.uci ?? "" } };
}

const REP_FLOOR = 0.25, REP_K = 5;

/** the positions since the last irreversible move, after this one (a capture or a man's move starts it again) */
export function repNext(rep: string[], st: CheckersState, m: CheckersMove, next: CheckersState): string[] {
  const k = C.key(next);
  if (m.cap >= 0 || Math.abs(st.sq[m.from]!) === 1) return [k];
  return [...rep, k];
}

function sideMaterial(st: CheckersState, side: number): number {
  let t = 0;
  for (const v of st.sq) if (v) t += (v > 0 ? 1 : -1) * (Math.abs(v) === 2 ? C.VAL.king : C.VAL.man);
  return side > 0 ? t : -t;
}

/** the no-repetition rule: the reading played, and a line saying so when it changed the move */
export function repRule(rd: MoveReading, p: ArrayLike<number>, st: CheckersState, rep: string[]): { rd: MoveReading; text: string | null } {
  if (!rd.move || !rd.state || rd.illegal) return { rd, text: null };
  if (!rep.includes(C.key(rd.state)) || sideMaterial(st, st.turn) < 0) return { rd, text: null };
  const w0 = Math.max(0, +(p[rd.cell] ?? 0)), fl = REP_FLOOR * w0;
  const hit = topLegal(st, p)
    .slice(0, REP_K)
    .find((c) => c.cell !== rd.cell && c.w >= fl && !rep.includes(C.key(C.make(st, c.m))));
  if (!hit) return { rd, text: null };
  const n = C.make(st, hit.m), cm = C.cellMove(hit.cell);
  return {
    rd: { ...rd, cell: hit.cell, move: hit.m, uci: C.uci(hit.m), state: n, from: cm.from, to: cm.to, score: hit.w, clean: false },
    text: `no repetition: ${rd.uci} would repeat a position, so ${C.uci(hit.m)} (${w0 > 0 ? ((100 * hit.w) / w0).toFixed(0) : "?"} % of its light) was played`,
  };
}
