// Whether the game is over: checkmate, stalemate, the 50-move rule, threefold repetition and insufficient material.
import type { ChessMove, ChessState } from "./position.js";
import { isLight } from "./position.js";
import { inCheck, key, moves } from "./moves.js";

export interface GameStatus {
  over: boolean;
  /** "1-0", "0-1", "1/2-1/2", or null while the game goes on */
  result: string | null;
  why: string | null;
  check: boolean;
  /** the legal moves, while the game goes on */
  moves?: ChessMove[];
}

/** the game's state; `keys` are the position keys seen so far (for threefold repetition), optional */
export function status(st: ChessState, keys?: string[]): GameStatus {
  const list = moves(st), check = inCheck(st);
  if (!list.length)
    return check ? { over: true, result: st.turn === "w" ? "0-1" : "1-0", why: "checkmate", check } : { over: true, result: "1/2-1/2", why: "stalemate", check };
  if (st.half >= 100) return { over: true, result: "1/2-1/2", why: "50-move rule", check };
  if (keys) {
    const k = key(st);
    let n = 0;
    for (const q of keys) if (q === k) n++;
    if (n >= 3) return { over: true, result: "1/2-1/2", why: "threefold repetition", check };
  }
  // the men besides the kings: none, a lone knight or bishop, or one bishop each on squares of the same color cannot mate
  const men: { p: number; sq: number }[] = [];
  for (let i = 0; i < 64; i++) if (st.sq[i] && Math.abs(st.sq[i]!) !== 6) men.push({ p: st.sq[i]!, sq: i });
  const a = men[0], b = men[1];
  const bare =
    men.length === 0 ||
    (men.length === 1 && (Math.abs(a!.p) === 2 || Math.abs(a!.p) === 3)) ||
    (men.length === 2 && Math.abs(a!.p) === 3 && Math.abs(b!.p) === 3 && Math.sign(a!.p) !== Math.sign(b!.p) && isLight(a!.sq) === isLight(b!.sq));
  if (bare) return { over: true, result: "1/2-1/2", why: "insufficient material", check };
  return { over: false, result: null, why: null, check, moves: list };
}
