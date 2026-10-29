// THE GHOST BOARD for checkers (the same ghosting as chess's ghost.ts, as it tries boards): the line of moves the look-ahead
// is reading now, played out from the position on the board — every piece that line moved, crowned or left standing somewhere new drawn
// faintly where it ends up, over the real board, which stays as it is. The screen draws what this returns; nothing here touches the game.
import type { CheckersMove, CheckersState } from "./board.js";
import { checkers as C } from "./board.js";

export interface CheckersGhost {
  /** a real square, and the piece the line leaves on it */
  sq: number;
  piece: number;
}

/** the pieces a line of moves put somewhere new: each square whose piece after the line differs from the board's, and is not empty */
export function checkersGhost(root: CheckersState, line: CheckersMove[]): CheckersGhost[] {
  let st = root;
  for (const m of line) st = C.make(st, m); // legal on the boards they were read on: replayed as given
  const out: CheckersGhost[] = [];
  for (let sq = 0; sq < 32; sq++) if (st.sq[sq] !== root.sq[sq] && st.sq[sq]) out.push({ sq, piece: st.sq[sq]! });
  return out;
}

/** how bright the ghost pieces are drawn (a lit pixel is 1): chess's 15 % */
export const CHECKERS_GHOST_LEVEL = 0.15;
