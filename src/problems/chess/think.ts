// THE CRYSTAL THINKS, AND THE FPGA'S RULES (the look-ahead's twin in the Python package: neuralcrystal.chess.look_ahead). The glass
// only ever answers "which move, and how good is this board for the side to move"; these round it.
//
// THINKING, WHEN UNSURE: when the brightest legal move holds under `share` of the top K's light, the crystal looks ahead. Widths W = [K, J,
// L, …], one per ply: our top K moves; below the last ply each board is solved (MIRRORED when the opponent is to move, so the crystal
// always sees the colour it was trained on) and its move map names the top W[d] legal replies — on the real board, un-mirrored — or, at
// the last ply, its value readout scores it: W + ½D on our turn, the opponent's L + ½D on theirs. Ours take the best child, theirs the
// worst; a move that ends the game scores directly (mate 1, any draw ½, mated 0). Ties keep the brighter. 2-ply 3 × 5: ≤ 18 reads.
//
// THE RULES, on the move chosen (the top K legal cells as the shortlist): take a mate in one; no stalemate when 3 or more up; no
// repetition (a position since the last irreversible move) when not behind — the brightest non-repeating candidate with at least
// `repFloor` of the chosen move's light, never a stalemate the rule above refuses.
import type { ChessMove, ChessReading, ChessState } from "./board.js";
import { chess as X, material, mirror } from "./board.js";
import { chessValueOf } from "./kit.js";

export interface ChessThinkOptions {
  /** "unsure": think only when the brightest holds under `share` of the top W[0]'s light; "always"; "off" */
  mode: "unsure" | "always" | "off";
  share: number;
  /** widths per ply: [3, 5] = 2-ply, our top 3 and their top 5 replies; [3] = 1-ply */
  widths: number[];
  /** how our candidates are ordered (orderCandidates) — the root only; the inner boards read by brightest cell as before */
  order?: CandidateOrder | null;
  /** the colour the crystal was trained to see on move (boards with the other to move are mirrored), or null */
  side: "w" | "b" | null;
  /** the crystal's raw light (all 4,096 cells) for a board: one read, four passes */
  look: (st: ChessState) => Promise<Float64Array>;
  moveP: (p: Float64Array) => Float64Array;
  valueCells: number[][];
  /** the game's position keys so far (threefold) */
  keys: string[];
  alive?: () => boolean;
  progress?: (done: number, total: number) => void;
  /**
   * NARROW THE SURE LINES (a tunable sureness per ply, for the future moves too): at
   * every node inside the tree, when the brightest of its top W[d] legal moves holds at least this share of their light, only that move is
   * followed — the depth stays the same, so every leaf is still judged the same distance ahead. `ours` on the crystal's plies, `theirs` on
   * the opponent's (a confident guess at your likeliest reply is not your most dangerous one: set it higher). 1 never narrows.
   */
  narrow?: { ours: number; theirs: number } | null;
  /** each board the look-ahead is about to read, as the line of moves that reaches it from the position thought about (the ghost board) */
  onLook?: (line: ChessMove[]) => void;
  /** each board's value as the crystal reads it once looked at: its expected score, W + ½D for the crystal (0 lost … 1 won) — the value meter */
  onValue?: (v: number, q: { w: number; d: number; l: number }) => void;
}

export interface ChessThought {
  thought: boolean;
  share: number;
  cand: { cell: number; move: ChessMove; w: number }[];
  es?: (number | null)[];
  best?: number;
  changed?: boolean;
  looked: number;
  /** the most reads the widths allow, and how many inner nodes were narrowed to their one sure move */
  most: number;
  narrowed: number;
  /** each inner board's sureness (its brightest move's share of its top W[d]), by ply d = 1, 2, …: what the thresholds are tuned against */
  shares: number[][];
  cancel?: boolean;
}

type Cand = { cell: number; move: ChessMove; w: number };

/**
 * HOW THE CANDIDATES ARE ORDERED (beyond the round robin, so that several of one piece's moves, say the queen's, can be tested).
 * brightest: by the cell's own light. piece: pieceOrder, round-robin — one move per piece
 * before any piece's second. blend: ONE normalised scale — a move's light is P(piece) × P(square | piece), with P(piece) = the piece's total
 * legal light ÷ (its legal moves)^gamma, normalised (gamma: the mobility correction — a queen gathers light by having twenty squares) and
 * P(square | piece) = the move's light ÷ its piece's total; score = log P(piece) + lambda · log P(square | piece): lambda 1 = the brightest
 * cell, lambda 0 = every move of the brightest piece before any other's; between, the piece that wants to move puts its 2nd and 3rd
 * squares ahead of other pieces' bests. maxPerPiece caps how many of the list's leading moves one piece may take (1 = round-robin-like
 * variety; the rest follow, so the list stays complete). floor: moves under this share of the brightest legal move's light go last
 * (a mobile piece's dimmest squares never make the shortlist).
 */
export interface CandidateOrder {
  mode: "brightest" | "piece" | "blend" | "guard";
  /** guard ("Brightest + top pieces"): how many of the most-lit pieces, and how many of each one's best squares, are guaranteed a place in the shortlist */
  guardPieces?: number;
  guardSquares?: number;
  lambda: number;
  gamma: number;
  maxPerPiece: number;
  floor: number;
}

export const CANDIDATE_ORDER: CandidateOrder = { mode: "brightest", lambda: 0.1, gamma: 0, maxPerPiece: 3, floor: 0.1, guardPieces: 1, guardSquares: 1 }; // on the save-the-piece boards, λ 0–0.1 saves the queen 80 %; the mobility correction γ 0.5 undoes it (33 %)

// Every ordered move carries `score`: its log-probability on the scale it was ranked by — log(light) for brightest, log P(piece) for piece
// first, the blend score for blend (a dim move under the floor: −∞) — so SURENESS can be read on that same scale (candidateShare).
export function orderCandidates<T extends { move: ChessMove; w: number; cell: number }>(cands: T[], o: CandidateOrder | null | undefined, k = 5): (T & { score: number })[] {
  const eps = 1e-12, all = [...cands].sort((a, b) => b.w - a.w || a.cell - b.cell).map((c) => ({ ...c, score: Math.log(Math.max(0, c.w) + eps) }));
  if (!o || o.mode === "brightest" || all.length < 2) return all;
  if (o.mode === "guard") {
    // BRIGHTEST + TOP PIECES (the piece holding the most light always gets its best move among the candidates): the brightest shortlist
    // of k, but the best guardSquares squares of each of the guardPieces pieces holding the most legal light are sure of a place — each missing one replaces the dimmest entry not itself guaranteed. [0] stays the
    // brightest cell, so the move without thinking is unchanged; piece-first reordering hurt ordinary play (a mobile queen gathers light
    // by having squares), this only makes sure a threatened piece's escapes get looked at.
    if (all.length <= k) return all;
    const tot = new Map<number, number>();
    for (const c of all) tot.set(c.move.from, (tot.get(c.move.from) ?? 0) + Math.max(0, c.w));
    const tops = [...tot.keys()].sort((a, b) => tot.get(b)! - tot.get(a)!).slice(0, Math.max(1, (o.guardPieces ?? 1) | 0));
    const must = tops.flatMap((f) => all.filter((c) => c.move.from === f).slice(0, Math.max(1, (o.guardSquares ?? 1) | 0)));
    const head = all.slice(0, k);
    for (const m of must) {
      if (head.includes(m)) continue;
      for (let j = head.length - 1; j > 0; j--) if (!must.includes(head[j]!)) { head[j] = m; break; }
    }
    return [...head, ...all.filter((c) => !head.includes(c))];
  }
  const top = Math.max(0, all[0]!.w), keep = all.filter((c) => c.w >= o.floor * top), dim = all.filter((c) => c.w < o.floor * top).map((c) => ({ ...c, score: -Infinity }));
  if (o.mode === "piece") {
    const pt = new Map<number, number>();
    let Zp = 0;
    for (const c of keep) { pt.set(c.move.from, (pt.get(c.move.from) ?? 0) + Math.max(0, c.w)); Zp += Math.max(0, c.w); }
    return [...pieceOrder(keep).map((c) => ({ ...c, score: Math.log((pt.get(c.move.from) ?? 0) / (Zp || 1) + eps) })), ...dim];
  }
  const tot = new Map<number, number>(), n = new Map<number, number>();
  for (const c of keep) { tot.set(c.move.from, (tot.get(c.move.from) ?? 0) + Math.max(0, c.w)); n.set(c.move.from, (n.get(c.move.from) ?? 0) + 1); }
  const adj = new Map<number, number>();
  let Z = 0;
  for (const [f, t] of tot) { const v = t / Math.pow(n.get(f)!, o.gamma); adj.set(f, v); Z += v; }
  const score = (c: T): number => Math.log(adj.get(c.move.from)! / (Z || 1) + eps) + o.lambda * Math.log(Math.max(0, c.w) / (tot.get(c.move.from) || 1) + eps);
  const ranked = keep.map((c) => ({ ...c, score: score(c) })).sort((a, b) => b.score - a.score || b.w - a.w);
  const cap = Math.max(1, o.maxPerPiece | 0), used = new Map<number, number>(), head: (T & { score: number })[] = [], later: (T & { score: number })[] = [];
  for (const c of ranked) { const u = used.get(c.move.from) ?? 0; if (u < cap) { head.push(c); used.set(c.move.from, u + 1); } else later.push(c); }
  return [...head, ...later, ...dim];
}

/**
 * SURENESS on the scale the candidates were ranked by (so it works with piece first and blend too): the first
 * candidate's share of exp(score) over the shortlist. Brightest: exactly the old share (its light ÷ the top K's). Blend at low lambda /
 * piece first: how sure the crystal is WHICH PIECE moves (then where) — the old light share under-read it, since the chosen piece's best
 * square is often dimmer than another piece's single bright move, and the unsure gate fired on nearly every move.
 */
export function candidateShare(cands: { score?: number; w: number }[]): number {
  if (!cands.length) return 1;
  if (cands.every((c) => c.score === undefined)) { const ls = cands.map((c) => Math.max(0, c.w)), t = ls.reduce((a, b) => a + b, 0); return t > 0 ? ls[0]! / t : 1; }
  const s = cands.map((c) => c.score ?? -Infinity), mx = Math.max(...s);
  if (!isFinite(mx)) return 1;
  const e = s.map((v) => Math.exp(v - mx)), t = e.reduce((a, b) => a + b, 0);
  return t > 0 ? e[0]! / t : 1;
}

function topLegal(st: ChessState, p: ArrayLike<number>, k: number, order?: CandidateOrder | null): (Cand & { score: number })[] {
  const all = [...X.moveCells(st).entries()].map(([cell, move]) => ({ cell, move, w: +(p[cell] ?? 0) }));
  return orderCandidates(all, order, k).slice(0, k);
}

/**
 * PIECE-FIRST: the move map spreads a piece's light over its destinations, so a
 * queen with twenty squares to flee to can hold far more light than a pawn and still lose the brightest-CELL read (in one probe,
 * 54 % of the light on queen moves vs 14 % on the pawn's, yet the brightest cell saved the pawn). The pieces are ranked by the SUM of their
 * legal moves' light, each piece's moves by their own light, and the list is dealt round-robin — every ranked piece's best move, then every
 * piece's second best — so [0] is the brightest piece's brightest move and a shortlist spans different pieces.
 * `cands` brightest first.
 */
export function pieceOrder<T extends { move: ChessMove; w: number }>(cands: T[]): T[] {
  const by = new Map<number, T[]>(), tot = new Map<number, number>();
  for (const c of cands) {
    const f = c.move.from;
    if (!by.has(f)) by.set(f, []);
    by.get(f)!.push(c);
    tot.set(f, (tot.get(f) ?? 0) + Math.max(0, c.w));
  }
  const order = [...by.keys()].sort((a, b) => tot.get(b)! - tot.get(a)!); // stable: equal totals keep the brighter piece first
  const out: T[] = [];
  for (let r = 0; out.length < cands.length; r++) for (const f of order) { const l = by.get(f)!; if (r < l.length) out.push(l[r]!); }
  return out;
}

/** the reading with the ordered list's first move in place of the brightest cell's (piece / blend order; a temperature draw is set aside) */
export function orderedReading(rd: ChessReading, st: ChessState, p: ArrayLike<number>, order: CandidateOrder): ChessReading & { pieceNote?: string } {
  if (order.mode === "brightest") return rd;
  const top = topLegal(st, p, 1, order)[0];
  if (!top) return rd;
  const m = top.move, moves = X.moves(st);
  if (rd.move && X.uci(rd.move) === X.uci(m)) return rd;
  return { ...rd, move: m, uci: X.uci(m), san: X.san(st, m, moves), state: X.make(st, m), from: m.from, to: m.to, cell: top.cell, score: top.w,
    pieceNote: `${order.mode === "piece" ? "piece-first" : `blend λ ${order.lambda} γ ${order.gamma}`}: ${X.san(st, m, moves)} instead of ${rd.move ? X.san(st, rd.move, moves) : "?"}` };
}

const CANCEL = { cancel: true };

/** st: the position (the crystal to move); pw: its MOVE light (value cells dark) */
export async function chessThink(st: ChessState, pw: ArrayLike<number>, o: ChessThinkOptions): Promise<ChessThought | null> {
  const W = o.widths.length ? o.widths : [3], K = Math.max(0, W[0]! | 0);
  if (o.mode === "off" || !K) return null;
  // the root's light is on the crystal's map: for the other color to move, that is the mirrored board's, so its moves are flipped back
  const rootSeen = o.side && st.turn !== o.side ? mirror(st) : st, rootLegal = X.moves(st);
  const cand = topLegal(rootSeen, pw, K, o.order)
    .map((c) => (rootSeen === st ? c : { ...c, move: rootLegal.find((x) => x.from === (c.move.from ^ 56) && x.to === (c.move.to ^ 56) && (x.promo || 0) === (c.move.promo || 0))! }))
    .filter((c) => !!c.move);
  const share = candidateShare(cand);
  const info: ChessThought = { thought: false, share, cand, looked: 0, most: 0, narrowed: 0, shares: [] };
  if (!(cand.length > 1 && (o.mode === "always" || share < o.share))) return info;
  info.thought = true;
  const me = st.turn, es: (number | null)[] = new Array(cand.length).fill(null), jobs: [number, ChessState, string[]][] = [];
  const endScore = (ov: { result?: string }): number => (ov.result === "1/2-1/2" ? 0.5 : (ov.result === "1-0") === (me === "w") ? 1 : 0);
  const orient = (n: ChessState): ChessState => (o.side && n.turn !== o.side ? mirror(n) : n);
  const dead = (): boolean => !!o.alive && !o.alive();
  cand.forEach((c, i) => {
    const n = X.make(st, c.move), k1 = [...o.keys, X.key(n)], ov = X.status(n, k1);
    if (ov.over) es[i] = endScore(ov);
    else jobs.push([i, n, k1]);
  });
  const P = W.length, S = new Array<number>(P + 1).fill(1);
  for (let d = P - 1; d >= 1; d--) S[d] = 1 + W[d]! * S[d + 1]!;
  let done = 0, total = jobs.length * (P > 1 ? S[1]! : 1);
  info.most = cand.length * (P > 1 ? S[1]! : 1);
  const tick = (): void => o.progress?.(++done, Math.max(done, total));
  const read = async (n: ChessState, line: ChessMove[]): Promise<{ raw: Float64Array; pos: ChessState; mir: boolean }> => {
    if (dead()) throw CANCEL;
    tick();
    o.onLook?.(line);
    const pos = orient(n), raw = await o.look(pos);
    info.looked++;
    if (o.onValue) {
      const q = chessValueOf(raw, o.valueCells);
      o.onValue(n.turn === me ? q.w + 0.5 * q.d : q.l + 0.5 * q.d, q); // q: the readout's own shares, for the side to move on that board
    }
    return { raw, pos, mir: pos !== n };
  };
  const val = async (n: ChessState, k: string[], d: number, line: ChessMove[]): Promise<number> => {
    const { raw, pos, mir } = await read(n, line);
    if (d === P) {
      const q = chessValueOf(raw, o.valueCells);
      return n.turn === me ? q.w + 0.5 * q.d : q.l + 0.5 * q.d;
    }
    const pm = o.moveP(raw), nm = X.moves(n), ours = n.turn === me, top = topLegal(pos, pm, W[d]!);
    let list = top
      .map((r) => (mir ? nm.find((x) => x.from === (r.move.from ^ 56) && x.to === (r.move.to ^ 56) && (x.promo || 0) === (r.move.promo || 0)) : r.move))
      .filter((m): m is ChessMove => !!m);
    const bar = o.narrow ? (ours ? o.narrow.ours : o.narrow.theirs) : 1;
    const lw = top.map((r) => Math.max(0, r.w)), sum = lw.reduce((a, b) => a + b, 0), sure = sum > 0 ? lw[0]! / sum : 1;
    if (list.length > 1) (info.shares[d] ??= []).push(sure);
    if (list.length > 1 && bar < 1) {
      if (sure >= bar) {
        list = list.slice(0, 1); // sure: only the brightest move is followed from here
        info.narrowed++;
      }
    }
    total -= (W[d]! - list.length) * S[d + 1]!;
    let best: number | null = null;
    for (const m of list) {
      const n2 = X.make(n, m), k2 = [...k, X.key(n2)], ov = X.status(n2, k2);
      let v: number;
      if (ov.over) {
        total -= S[d + 1]!;
        v = endScore(ov);
      } else v = await val(n2, k2, d + 1, [...line, m]);
      if (best === null || (ours ? v > best : v < best)) best = v;
    }
    if (best === null) {
      const q = chessValueOf(raw, o.valueCells);
      return n.turn === me ? q.w + 0.5 * q.d : q.l + 0.5 * q.d;
    }
    return best;
  };
  try {
    for (const [i, n, k1] of jobs) {
      if (P === 1) {
        // 1-ply: the opponent's own W/D/L right after our move — their loss is our win
        const { raw } = await read(n, [cand[i]!.move]), q = chessValueOf(raw, o.valueCells);
        es[i] = q.l + 0.5 * q.d;
      } else es[i] = await val(n, k1, 1, [cand[i]!.move]);
    }
  } catch (e) {
    if (e === CANCEL) return Object.assign(info, { cancel: true });
    throw e;
  }
  if (dead()) return Object.assign(info, { cancel: true });
  let best = 0;
  for (let i = 1; i < es.length; i++) if ((es[i] ?? -1) > (es[best] ?? -1)) best = i; // strictly better only: a tie keeps the brighter
  return Object.assign(info, { es, best, changed: best !== 0 });
}

/** the reading after thinking: the candidate it chose (any temperature draw is set aside once the crystal has thought) */
export function thoughtReading(rd: ChessReading, st: ChessState, th: ChessThought | null): ChessReading & { thoughtNote?: string } {
  if (!th || !th.thought || th.cancel || th.best === undefined) return rd;
  const c = th.cand[th.best]!, m = c.move, moves = X.moves(st);
  // how sure the inner boards were, ply by ply: the median and the most (ply 1 is our move; the inner boards start with your reply, ply 2)
  const med = (a: number[]): number => [...a].sort((x, y) => x - y)[a.length >> 1]!;
  const sure = th.shares
    .map((a, d) => (a && a.length ? `ply ${d + 1} (${d % 2 ? "yours" : "ours"}) ${med(a).toFixed(2)}–${Math.max(...a).toFixed(2)}` : ""))
    .filter(Boolean)
    .join(", ");
  const note = `looked ahead: ${th.cand.map((q, i) => `${X.san(st, q.move, moves)} ${(th.es?.[i] ?? 0).toFixed(2)}`).join(", ")} (${th.looked} of up to ${th.most} reads${th.narrowed ? `, ${th.narrowed} sure lines narrowed` : ""})${sure ? ` · sureness, median–most: ${sure}` : ""}`;
  if (c.cell === rd.cell && rd.move && X.uci(rd.move) === X.uci(m)) return { ...rd, thoughtNote: note };
  return { ...rd, move: m, uci: X.uci(m), san: X.san(st, m, moves), state: X.make(st, m), from: m.from, to: m.to, cell: c.cell, score: c.w, thoughtNote: note };
}

export interface ChessRules {
  mate: boolean;
  stale: boolean;
  rep: boolean;
  /** the replacement's light, as a share of the chosen move's */
  repFloor: number;
  /** the shortlist */
  topK: number;
}

export const CHESS_RULES: ChessRules = { mate: true, stale: true, rep: true, repFloor: 0.25, topK: 3 };

/** the positions since the last irreversible move after m (cleared by a pawn move, a capture or a castling-rights change) */
export function repNext(rep: string[], prev: ChessState, m: ChessMove, next: ChessState): string[] {
  const k = X.key(next), c0 = prev.castle, c1 = next.castle;
  const irr = Math.abs(m.piece || prev.sq[m.from]!) === 1 || !!m.capture || c0.K !== c1.K || c0.Q !== c1.Q || c0.k !== c1.k || c0.q !== c1.q;
  return irr ? [k] : [...rep, k];
}

/** the FPGA's rules on the chosen move: rd unchanged, or the replacement with `ruled` saying why */
export function chessRules(rd: ChessReading, p: ArrayLike<number>, st: ChessState, rep: string[], R: ChessRules = CHESS_RULES): ChessReading & { ruled?: string } {
  if (!rd.move || rd.illegal || !(R.mate || R.stale || R.rep)) return rd;
  const side = st.turn, sgn = side === "w" ? 1 : -1, cells = X.moveCells(st);
  const cand = [...cells.entries()].map(([cell, move]) => ({ cell, move, w: Math.max(0, +(p[cell] ?? 0)) })).sort((a, b) => b.w - a.w || a.cell - b.cell).slice(0, Math.max(1, R.topK | 0));
  const info = new Map<number, { n: ChessState; mate: boolean; stale: boolean; rep: boolean; mat: number }>();
  const look = (c: Cand): { n: ChessState; mate: boolean; stale: boolean; rep: boolean; mat: number } => {
    let o = info.get(c.cell);
    if (!o) {
      const n = X.make(st, c.move), none = !X.moves(n).length, chk = X.inCheck(n);
      o = { n, mate: none && chk, stale: none && !chk, rep: rep.includes(X.key(n)), mat: sgn * material(n) };
      info.set(c.cell, o);
    }
    return o;
  };
  const cell0 = rd.cell ?? X.MOVE_CELL(rd.move), orig: Cand = { cell: cell0, move: cells.get(cell0) ?? rd.move, w: Math.max(0, +(p[cell0] ?? 0)) };
  const staleBad = (c: Cand): boolean => {
    const o = look(c);
    return o.stale && o.mat >= 3;
  };
  let cur = orig;
  const fired: string[] = [];
  if (R.mate && !look(cur).mate) {
    const hit = cand.find((c) => look(c).mate);
    if (hit) {
      cur = hit;
      fired.push("take mate in 1");
    }
  }
  if (R.stale && staleBad(cur)) {
    const hit = cand.find((c) => !look(c).stale);
    if (hit) {
      cur = hit;
      fired.push("no stalemate when ahead");
    }
  }
  if (R.rep && look(cur).rep && sgn * material(st) >= 0) {
    const fl = Math.max(0, R.repFloor) * cur.w, hit = cand.find((c) => c.cell !== cur.cell && !look(c).rep && c.w >= fl && !(R.stale && staleBad(c)));
    if (hit) {
      cur = hit;
      fired.push("no repetition");
    }
  }
  if (cur.cell === orig.cell) return rd;
  const m = cur.move, moves = X.moves(st), san = X.san(st, m, moves);
  return { ...rd, move: m, uci: X.uci(m), san, state: look(cur).n, from: m.from, to: m.to, cell: cur.cell, score: cur.w, ruled: `${fired.join(" + ")}: played ${san} instead of ${rd.san}` };
}
