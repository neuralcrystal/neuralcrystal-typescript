// The rules of movement: attacks, the moves the pieces allow, making a move, the legal list, and the move written as SAN and UCI.
import type { ChessMove, ChessState } from "./position.js";
import { copy, PIECE, pieceChar, sqName } from "./position.js";

type Dirs = readonly (readonly [number, number])[];
const KN: Dirs = [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]];
const KG: Dirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
const BI: Dirs = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
const RO: Dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const PROMOS = [5, 2, 4, 3];

/** file and rank both on the board */
export const inb = (f: number, r: number): boolean => f >= 0 && f < 8 && r >= 0 && r < 8;

/** is sq attacked by color `by` (+1 white, −1 black)? */
export function attacked(sq: number, by: number, board: Int8Array): boolean {
  const f = sq & 7, r = sq >> 3;
  // a pawn of `by` one rank towards its own side, on a neighboring file
  const pr = r - by;
  if (pr >= 0 && pr < 8) {
    for (const df of [-1, 1]) {
      const pf = f + df;
      if (inb(pf, pr) && board[pr * 8 + pf] === by) return true;
    }
  }
  for (const [df, dr] of KN) {
    const F = f + df, R = r + dr;
    if (inb(F, R) && board[R * 8 + F] === 2 * by) return true;
  }
  for (const [df, dr] of KG) {
    const F = f + df, R = r + dr;
    if (inb(F, R) && board[R * 8 + F] === 6 * by) return true;
  }
  const ray = (dirs: Dirs, kind: number): boolean => {
    for (const [df, dr] of dirs) {
      let F = f + df, R = r + dr;
      while (inb(F, R)) {
        const p = board[R * 8 + F]!;
        if (p) {
          if (p === kind * by || p === 5 * by) return true;
          break;
        }
        F += df;
        R += dr;
      }
    }
    return false;
  };
  return ray(BI, 3) || ray(RO, 4);
}

export const kingSq = (board: Int8Array, col: number): number => board.indexOf(6 * col);

export function inCheck(st: ChessState): boolean {
  const col = st.turn === "w" ? 1 : -1, k = kingSq(st.sq, col);
  return k >= 0 && attacked(k, -col, st.sq);
}

/** the moves the pieces allow, before the rule that a move may not leave its own king in check */
export function pseudo(st: ChessState): ChessMove[] {
  const col = st.turn === "w" ? 1 : -1, b = st.sq, out: ChessMove[] = [];
  const add = (from: number, to: number, extra?: Partial<ChessMove>): void => {
    out.push({ from, to, piece: b[from]!, capture: b[to] !== 0, ...extra });
  };
  for (let from = 0; from < 64; from++) {
    const p = b[from]!;
    if (!p || Math.sign(p) !== col) continue;
    const f = from & 7, r = from >> 3, kind = Math.abs(p);
    if (kind === 1) {
      const r1 = r + col, start = col > 0 ? 1 : 6, last = col > 0 ? 7 : 0;
      if (inb(f, r1) && !b[r1 * 8 + f]) {
        const to = r1 * 8 + f;
        if (to >> 3 === last) for (const promo of PROMOS) add(from, to, { promo });
        else add(from, to);
        const r2 = r + 2 * col;
        if (r === start && !b[r2 * 8 + f]) add(from, r2 * 8 + f, { double: true });
      }
      for (const df of [-1, 1]) {
        const F = f + df;
        if (!inb(F, r1)) continue;
        const to = r1 * 8 + F, q = b[to]!;
        if (q && Math.sign(q) === -col) {
          if (to >> 3 === last) for (const promo of PROMOS) add(from, to, { promo, capture: true });
          else add(from, to, { capture: true });
        } else if (to === st.ep && !q) add(from, to, { ep: true, capture: true });
      }
      continue;
    }
    const dirs = kind === 2 ? KN : kind === 3 ? BI : kind === 4 ? RO : KG, slide = kind >= 3 && kind <= 5;
    for (const [df, dr] of dirs) {
      let F = f + df, R = r + dr;
      while (inb(F, R)) {
        const to = R * 8 + F, q = b[to]!;
        if (q) {
          if (Math.sign(q) === -col) add(from, to);
          break;
        }
        add(from, to);
        if (!slide) break;
        F += df;
        R += dr;
      }
    }
    if (kind === 6) {
      // castling: the right, the squares between empty, not in check, and the king's path not attacked (the rook's square may be)
      const home = col > 0 ? 4 : 60;
      if (from !== home) continue;
      const ks = col > 0 ? st.castle.K : st.castle.k, qs = col > 0 ? st.castle.Q : st.castle.q;
      const safe = (d: number): boolean => !attacked(home + d, -col, b);
      if (ks && !b[home + 1] && !b[home + 2] && b[home + 3] === 4 * col && safe(0) && safe(1) && safe(2)) add(from, home + 2, { castle: "K" });
      if (qs && !b[home - 1] && !b[home - 2] && !b[home - 3] && b[home - 4] === 4 * col && safe(0) && safe(-1) && safe(-2)) add(from, home - 2, { castle: "Q" });
    }
  }
  return out;
}

/** the position after a move (a new state) */
export function make(st: ChessState, m: ChessMove): ChessState {
  const n = copy(st), b = n.sq, col = st.turn === "w" ? 1 : -1, p = b[m.from]!, kind = Math.abs(p);
  n.half = kind === 1 || b[m.to] ? 0 : st.half + 1;
  n.ep = -1;
  b[m.to] = m.promo ? m.promo * col : p;
  b[m.from] = 0;
  if (m.ep) b[m.to - 8 * col] = 0;
  if (m.castle === "K") {
    b[m.from + 1] = b[m.from + 3]!;
    b[m.from + 3] = 0;
  }
  if (m.castle === "Q") {
    b[m.from - 1] = b[m.from - 4]!;
    b[m.from - 4] = 0;
  }
  if (m.double) {
    // en passant is offered only when an enemy pawn stands beside (whether taking it is legal is checked by the move list)
    const f = m.to & 7, r = m.to >> 3;
    for (const df of [-1, 1]) if (inb(f + df, r) && b[r * 8 + f + df] === -col) n.ep = m.to - 8 * col;
  }
  if (kind === 6) {
    if (col > 0) n.castle.K = n.castle.Q = false;
    else n.castle.k = n.castle.q = false;
  }
  const rookHome = (sq: number, c: number): void => {
    if (c > 0) {
      if (sq === 7) n.castle.K = false;
      if (sq === 0) n.castle.Q = false;
    } else {
      if (sq === 63) n.castle.k = false;
      if (sq === 56) n.castle.q = false;
    }
  };
  if (kind === 4) rookHome(m.from, col);
  rookHome(m.to, -col); // a rook taken on its home square takes that right with it
  n.turn = col > 0 ? "b" : "w";
  if (col < 0) n.full = st.full + 1;
  return n;
}

/** every legal move */
export function moves(st: ChessState): ChessMove[] {
  const col = st.turn === "w" ? 1 : -1, out: ChessMove[] = [];
  for (const m of pseudo(st)) {
    const n = make(st, m), k = kingSq(n.sq, col);
    if (k >= 0 && !attacked(k, -col, n.sq)) out.push(m);
  }
  return out;
}

/** the position in FEN; the en-passant square is written only when the capture is legal (python-chess's "legal" style) */
export function fen(st: ChessState): string {
  let s = "";
  for (let r = 7; r >= 0; r--) {
    let e = 0;
    for (let f = 0; f < 8; f++) {
      const p = st.sq[r * 8 + f]!;
      if (!p) {
        e++;
        continue;
      }
      if (e) {
        s += e;
        e = 0;
      }
      s += pieceChar(p);
    }
    if (e) s += e;
    if (r) s += "/";
  }
  const cs = (st.castle.K ? "K" : "") + (st.castle.Q ? "Q" : "") + (st.castle.k ? "k" : "") + (st.castle.q ? "q" : "");
  return `${s} ${st.turn} ${cs || "-"} ${st.ep >= 0 && moves(st).some((m) => m.ep) ? sqName(st.ep) : "-"} ${st.half} ${st.full}`;
}

/** the position's identity for repetition: the FEN without its move counters */
export const key = (st: ChessState): string => fen(st).split(" ").slice(0, 4).join(" ");

export function uci(m: ChessMove): string;
export function uci(m: ChessMove | null): string | null;
export function uci(m: ChessMove | null): string | null {
  // a null move has no name (the original read straight through it and threw; nothing passes one)
  if (!m) return null;
  return sqName(m.from) + sqName(m.to) + (m.promo ? PIECE[m.promo - 1]!.toLowerCase() : "");
}

export const fromUci = (st: ChessState, u: string): ChessMove | null => moves(st).find((m) => uci(m) === u) || null;

/** the move in standard algebraic notation; `list` is the legal list, if already known */
export function san(st: ChessState, m: ChessMove, list?: ChessMove[]): string {
  const all = list || moves(st), n = make(st, m), kind = Math.abs(m.piece);
  let s: string;
  if (m.castle) s = m.castle === "K" ? "O-O" : "O-O-O";
  else {
    if (kind === 1) s = m.capture ? "abcdefgh"[m.from & 7] + "x" : "";
    else {
      s = PIECE[kind - 1]!;
      const others = all.filter((o) => (o.from !== m.from || o.to !== m.to || o.promo !== m.promo) && o.piece === m.piece && o.to === m.to);
      if (others.length) {
        const sameFile = others.some((o) => (o.from & 7) === (m.from & 7)), sameRank = others.some((o) => o.from >> 3 === m.from >> 3);
        s += !sameFile ? "abcdefgh"[m.from & 7] : !sameRank ? String(1 + (m.from >> 3)) : sqName(m.from);
      }
      if (m.capture) s += "x";
    }
    s += sqName(m.to);
    if (m.promo) s += "=" + PIECE[m.promo - 1];
  }
  if (inCheck(n)) s += moves(n).length ? "+" : "#";
  return s;
}
