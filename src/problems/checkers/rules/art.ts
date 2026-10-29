// The piece art, one set per board layout. A piece is a square glyph of "#" (lit) and "." (dark). The side to move is a solid disc and
// the opponent a ring; a king of either carries a small crown, cut dark out of the disc and lit inside the ring. The crown is the only
// difference between a man and a king, so it is the one feature the optics has to resolve.

export interface GlyphSet {
  man: string[];
  king: string[];
  foe: string[];
  foek: string[];
}

/** a cell layout: the board cell, the bracket's arm and stroke, the glyph and its inset, the jump box, and the art */
export interface Layout {
  TILE: number;
  ARM: number;
  SPR: number;
  INS: number;
  STROKE: number;
  BOX: number;
  /** the jump box sits evenly across the cell edge (the 32-unit cell) instead of mostly inside it */
  BOXEVEN: boolean;
  GLYPH: GlyphSet;
  /** the art on the square that must go on jumping (a smaller piece, so the box around it has room) */
  MARK: GlyphSet;
}

// The 8 × 8 glyphs of the 12-unit cell. The crown is four wide and three tall, centered, two spikes over a bar: all the shape that
// survives at this pitch.
const GLYPH12: GlyphSet = {
  man: ["..####..", ".######.", "########", "########", "########", "########", ".######.", "..####.."],
  king: ["..####..", ".######.", "##.##.##", "##....##", "##....##", "########", ".######.", "..####.."],
  foe: ["..####..", ".#....#.", "#......#", "#......#", "#......#", "#......#", ".#....#.", "..####.."],
  foek: ["..####..", ".#....#.", "#.#..#.#", "#.####.#", "#.####.#", "#......#", ".#....#.", "..####.."],
};

// The 7 × 7 glyphs for the square that must jump again, which carries a 2-pixel box: the 8-pixel piece would touch it. Only the side to
// move ever jumps, so the canonical frame only needs the solid pieces; the rings are here for screens that keep each side its own color.
const GLYPH7_12: GlyphSet = {
  man: ["..###..", ".#####.", "#######", "#######", "#######", ".#####.", "..###.."],
  king: ["..###..", ".#####.", "##.#.##", "##...##", "##...##", ".#####.", "..###.."],
  foe: ["..###..", ".#...#.", "#.....#", "#.....#", "#.....#", ".#...#.", "..###.."],
  foek: ["..###..", ".#...#.", "#.#.#.#", "#.###.#", "#.....#", ".#...#.", "..###.."],
};

// The 20 × 20 ring of the 32-unit cell, hand-drawn: 3 pixels on the sides with a mixed 1- and 2-pixel step on the diagonals.
const FOE32 = [
  ".....##########.....", ".....##########.....", "..################..", "..###..........###..", "..###..........###..",
  "###..............###", "###..............###", "###..............###", "###..............###", "###..............###",
  "###..............###", "###..............###", "###..............###", "###..............###", "###..............###",
  "..###..........###..", "..###..........###..", "..################..", ".....##########.....", ".....##########.....",
];

// The outlined king of the 32-unit cell: the ring with a 10-pixel crown at cols 5–14, rows 7–12 (three equal 2-pixel points over a
// 3-row band). The crown is whatever it lights that the ring does not.
const FOEK32 = [
  ".....##########.....", ".....##########.....", "..################..", "..###..........###..", "..###..........###..",
  "###..............###", "###..............###", "###..##..##..##..###", "###..##..##..##..###", "###..##..##..##..###",
  "###..##########..###", "###..##########..###", "###..##########..###", "###..............###", "###..............###",
  "..###..........###..", "..###..........###..", "..################..", ".....##########.....", ".....##########.....",
];

/** nearest-neighbor upscale of every glyph in a set: one pixel becomes a k × k block, so the pixel-art look survives exactly */
export function upscale(art: GlyphSet, k: number): GlyphSet {
  const up = (g: string[]): string[] => {
    const n = g.length, rows: string[] = [];
    for (let y = 0; y < n * k; y++) {
      let row = "";
      for (let x = 0; x < n * k; x++) row += g[(y / k) | 0]![(x / k) | 0];
      rows.push(row);
    }
    return rows;
  };
  return { man: up(art.man), king: up(art.king), foe: up(art.foe), foek: up(art.foek) };
}

/** the whole set from a ring and its outlined king: the man is the ring with its hole filled, the solid king has the crown cut dark */
export function fromRing(foe: string[], foek: string[]): GlyphSet {
  const man = foe.map((r) => {
    const a = r.indexOf("#"), b = r.lastIndexOf("#");
    return a < 0 ? r : r.slice(0, a) + "#".repeat(b - a + 1) + r.slice(b + 1);
  });
  const king = man.map((r, y) => [...r].map((ch, x) => (foek[y]![x] === "#" && foe[y]![x] !== "#" ? "." : ch)).join(""));
  return { man, king, foe, foek };
}

const SET32 = fromRing(FOE32, FOEK32);

// Three layouts, each filling a frame of 8 cells exactly: 96, 192 and 256 pixels. The 24-unit cell is the 12-unit one doubled, constant
// for constant. The 32-unit cell lines up with lenslet arrays of 16, 8, 4 and 2 across the frame, so one lenslet of the second stage
// covers exactly one board square. Its dark gap between bracket and piece scales with the cell (inset 6, not 4), and its marked piece
// stays full size, since the even box already leaves 3 dark pixels on every side.
export const LAYOUTS: Record<number, Layout> = {
  12: { TILE: 12, ARM: 11, SPR: 8, INS: 2, STROKE: 1, BOX: 2, BOXEVEN: false, GLYPH: GLYPH12, MARK: GLYPH7_12 },
  24: { TILE: 24, ARM: 22, SPR: 16, INS: 4, STROKE: 2, BOX: 4, BOXEVEN: false, GLYPH: upscale(GLYPH12, 2), MARK: upscale(GLYPH7_12, 2) },
  32: { TILE: 32, ARM: 29, SPR: 20, INS: 6, STROKE: 3, BOX: 5, BOXEVEN: true, GLYPH: SET32, MARK: SET32 },
};
