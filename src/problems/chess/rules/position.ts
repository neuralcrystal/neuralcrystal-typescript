// The symbolic position: sq[64] (a1 = 0 … h8 = 63; +1..+6 white P N B R Q K, negative black), the side to move, the castling rights,
// the en-passant square and the move counters. Parsing, square names and a small seeded random source.

export interface ChessState {
  sq: Int8Array;
  turn: "w" | "b";
  castle: { K: boolean; Q: boolean; k: boolean; q: boolean };
  /** the en-passant target square, or −1 */
  ep: number;
  half: number;
  full: number;
}

export interface ChessMove {
  from: number;
  to: number;
  /** the promotion piece type (2..5: N B R Q), or 0 / undefined */
  promo?: number;
  piece: number;
  capture?: boolean;
  castle?: "K" | "Q";
  ep?: boolean;
  double?: boolean;
}

/** the piece letters, type 1..6 */
export const PIECE = "PNBRQK";

/** mulberry32: a seeded generator of numbers in [0, 1) */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const sqName = (sq: number): string => "abcdefgh"[sq & 7]! + (1 + (sq >> 3));
export const sqIndex = (s: string): number => s.charCodeAt(0) - 97 + 8 * (s.charCodeAt(1) - 49);
export const pieceChar = (p: number): string => (p > 0 ? PIECE[p - 1]! : p < 0 ? PIECE[-p - 1]!.toLowerCase() : ".");
/** a1 dark, h1 light */
export const isLight = (sq: number): boolean => (((sq & 7) + (sq >> 3)) & 1) === 1;
/** 0 empty, 1..6 white P N B R Q K, 7..12 black */
export const classOf = (p: number): number => (p === 0 ? 0 : p > 0 ? p : 6 - p);
export const pieceOfClass = (c: number): number => (c === 0 ? 0 : c <= 6 ? c : -(c - 6));

export function parseFEN(fen: string): ChessState {
  const f = fen.trim().split(/\s+/), sq = new Int8Array(64), rows = f[0]!.split("/");
  if (rows.length !== 8) throw new Error("bad FEN: " + fen);
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]!) {
      if (ch >= "1" && ch <= "8") {
        file += +ch;
        continue;
      }
      const i = PIECE.indexOf(ch.toUpperCase());
      if (i < 0) throw new Error("bad FEN piece " + ch);
      sq[(7 - r) * 8 + file] = ch === ch.toUpperCase() ? i + 1 : -(i + 1);
      file++;
    }
  }
  const cs = f[2] || "-";
  return {
    sq,
    turn: f[1] === "b" ? "b" : "w",
    castle: { K: cs.includes("K"), Q: cs.includes("Q"), k: cs.includes("k"), q: cs.includes("q") },
    ep: f[3] && f[3] !== "-" ? sqIndex(f[3]) : -1,
    half: +(f[4] || 0),
    full: +(f[5] || 1),
  };
}

export const start = (): ChessState => parseFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

export const copy = (st: ChessState): ChessState => ({ sq: Int8Array.from(st.sq), turn: st.turn, castle: { ...st.castle }, ep: st.ep, half: st.half, full: st.full });

/** the squares whose piece differs between two positions */
export function changed(a: ChessState, b: ChessState): number[] {
  const out: number[] = [];
  for (let i = 0; i < 64; i++) if (a.sq[i] !== b.sq[i]) out.push(i);
  return out;
}

/** python-chess Board.mirror(): a1 ↔ a8, colors swapped, the other side to move, K ↔ k / Q ↔ q, the ep square flipped */
export function mirror(st: ChessState): ChessState {
  const n = copy(st), s = new Int8Array(64);
  for (let i = 0; i < 64; i++) s[i ^ 56] = -st.sq[i]!;
  n.sq = s;
  n.turn = st.turn === "w" ? "b" : "w";
  n.castle = { K: !!st.castle.k, Q: !!st.castle.q, k: !!st.castle.K, q: !!st.castle.Q };
  n.ep = st.ep >= 0 ? st.ep ^ 56 : -1;
  return n;
}

/** A move on a position → the same move on its mirror (the Python package's mirror_move): squares flipped a1 ↔ a8, the piece's color swapped. */
export function mirrorMove(m: ChessMove): ChessMove {
  return { ...m, from: m.from ^ 56, to: m.to ^ 56, piece: -m.piece };
}
