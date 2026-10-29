// The symbols and placements the Craftax board picture is drawn from: 16 × 16 glyphs for every block, item, creature and player facing, the
// picture's fixed pixels, and where each status pad's mirrors sit. The data ships with the module (atlas.json, the same file as the Python
// package's craftax_frame_atlas.json, compiled into atlas.gen.ts), so it loads the same way in Node and the browser, with no fetch.
import { ATLAS_JSON } from "./atlas.gen.js";

/** atlas.json as stored: bit planes as base64, the rest plain */
export interface CraftaxAtlasJson {
  /** the picture's side, F (192) */
  frame: number;
  /** a glyph's side (16) */
  tile: number;
  /** the view around the player, [rows, cols] (9 × 11) */
  view: [number, number];
  n_blocks: number;
  n_items: number;
  /** creature classes, in drawing order */
  classes: string[];
  n_glyphs: number;
  /** all glyphs' bits, high bit first, glyph after glyph */
  glyphs: string;
  /** each class's first glyph per creature type (projectiles: four glyphs each, one per heading) */
  cls_base: number[][];
  /** the player's first glyph (four facings follow) */
  player0: number;
  /** the glyph of an unlit tile */
  dark: number;
  bars: boolean;
  /** the picture's fixed pixels, F × F bits */
  static: string;
  /** the status rows' names */
  hud: string[];
  /** per status row, its 9 pads, each a list of pixel indices */
  pads: number[][][];
  modes: string[];
  blocks: string[];
  items: string[];
}

/** the atlas ready to draw from: the bit planes unpacked */
export interface CraftaxAtlas extends CraftaxAtlasJson {
  /** n_glyphs × tile × tile, 0 / 1 */
  glyphBits: Uint8Array;
  /** frame × frame, 0 / 1 */
  staticBits: Uint8Array;
}

/** base64 → n bits, high bit first (atob is global in browsers and Node) */
export function unpackBits(b64: string, n: number): Uint8Array {
  const s = atob(b64), out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (s.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
  return out;
}

let cached: CraftaxAtlas | null = null;

/** The atlas, unpacked once and shared. */
export function craftaxAtlas(): CraftaxAtlas {
  if (!cached) {
    const A = ATLAS_JSON as unknown as CraftaxAtlasJson, TL = A.tile;
    cached = { ...A, glyphBits: unpackBits(A.glyphs, A.n_glyphs * TL * TL), staticBits: unpackBits(A.static, A.frame * A.frame) };
  }
  return cached;
}
