// THE GHOST BOARD: while the crystal looks ahead, the line of moves it is reading now, played out from the position on the board — every
// piece that line moved (or promoted) drawn as a faint (15 %) ghost where it ends up, over the real board, which stays as it is. The
// screen draws what this returns; nothing here touches the game.
import type { ChessMove, ChessState } from "./board.js";
import { chess as C } from "./board.js";

export interface GhostPiece {
  sq: number;
  piece: number;
}

/** the pieces a line of moves put somewhere new: each square whose piece after the line differs from the board's, and is not empty */
export function ghostPieces(root: ChessState, line: ChessMove[]): GhostPiece[] {
  let st = root;
  for (const m of line) {
    // the line's moves are legal on the boards they were read on; replay them as given
    st = C.make(st, m);
  }
  const out: GhostPiece[] = [];
  for (let sq = 0; sq < 64; sq++) if (st.sq[sq] !== root.sq[sq] && st.sq[sq]) out.push({ sq, piece: st.sq[sq]! });
  return out;
}

/** how bright the ghost pieces are drawn (a lit pixel is 1) */
export const GHOST_LEVEL = 0.15;
