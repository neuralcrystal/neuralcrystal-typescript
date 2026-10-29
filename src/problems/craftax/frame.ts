// The Craftax crystal's board picture: the 192 × 192 mirror frame drawn from a game state. The 9 × 11 tiles around the player as 16 × 16
// glyphs (the creature, else the item, else the block; unlit tiles dark; the player turned to its facing; fighters with a health bar),
// the status pads under them, and, optionally, the player's tile marked with its needs and its pickaxe. Memory cells are left blank:
// pictures() writes the memory into the second input picture.
import type { CraftaxAtlas } from "./atlas.js";
import { craftaxAtlas } from "./atlas.js";
import type { CraftaxMob, CraftaxState } from "./state.js";

/** light (0 … 255) above this is lit: the JAX game's light > 0.05 */
const VISIBLE = 12;
const CLASS_FIELD: Record<string, keyof CraftaxState> = {
  passive: "passive", melee: "melee", ranged: "ranged", "mob projectile": "mob_projectiles", "your projectile": "player_projectiles",
};

/** round halves to even, as JAX and Python do */
export function roundEven(x: number): number {
  const f = Math.floor(x), d = x - f;
  return d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f;
}

/** which of each status row's 9 pads are lit, in the atlas's hud order */
function hudOn(A: CraftaxAtlas, s: CraftaxState, mode: number): boolean[][] {
  const inv = s.inventory, lv = s.level;
  const thermo = (n: number): boolean[] => Array.from({ length: 9 }, (_, k) => k < Math.min(9, n));
  const root = (v: number): boolean[] => thermo(roundEven(Math.sqrt(v)));
  const hot = (i: number): boolean[] => Array.from({ length: 9 }, (_, k) => k === i);
  const v: Record<string, boolean[]> = {
    health: thermo(roundEven(s.health)), food: thermo(s.food), drink: thermo(s.drink), energy: thermo(s.energy), mana: thermo(s.mana),
    xp: thermo(s.xp), dexterity: thermo(s.dexterity), strength: thermo(s.strength), intelligence: thermo(s.intelligence),
    floor: hot(lv), light: thermo(lv === 0 ? roundEven(9 * s.daylight) : 4),
    "pickaxe tier": thermo(inv.pickaxe), "sword tier": thermo(inv.sword), bow: thermo(inv.bow), books: thermo(inv.books),
    "sword ench.": thermo(s.sword_enchant), "bow ench.": thermo(s.bow_enchant),
    facing: hot([3, 5, 1, 7][Math.min(3, Math.max(0, s.dir - 1))]!), mode: hot([0, 2, 6, 8][mode]!),
  };
  for (const n of ["wood", "stone", "coal", "iron", "diamond", "sapling", "ruby", "sapphire", "arrows", "torches"] as const) v[n] = root(inv[n]);
  ["red", "green", "blue", "pink", "cyan", "yellow"].forEach((c, i) => (v[`potion ${c}`] = root(inv.potions[i] ?? 0)));
  ["helmet", "chestplate", "pants", "boots"].forEach((n, i) => (v[n] = thermo(inv.armour[i] ?? 0)));
  ["helmet ench.", "chest ench.", "pants ench.", "boots ench."].forEach((n, i) => (v[n] = thermo(s.armour_enchant[i] ?? 0)));
  const boss = lv === 8 && s.melee.length === 0 && s.ranged.length === 0 && s.boss_spawn_turns <= 0;
  v["flags"] = [s.sleeping, s.resting, !!s.spells[0], !!s.spells[1], s.monsters_killed >= 8, boss, false, false, false];
  return A.hud.map((n) => v[n]!);
}

// The player's tile mark (the crystals were trained with it): the 16 × 16 player square at rows 64..79, cols 80..95. Each need at 3 or
// less hollows part of the body (thirsty: head; hungry: tummy; tired: arms; hurt: legs and feet), then the pickaxe tier lights 3 × 3
// corners 1 px in: top right (wood), top left (stone), bottom left (iron), bottom right (diamond).
const TILE_Y = 64, TILE_X = 80;
const HOLES = [[[5, 7], [5, 8], [6, 7], [6, 8]], [[8, 7], [8, 8], [9, 7], [9, 8]], [[8, 4], [8, 5], [9, 4], [8, 10], [8, 11], [9, 11]],
  [[10, 6], [10, 9], [11, 5], [11, 6], [11, 9], [11, 10]]] as const;
const CORNERS = [[1, 12], [1, 1], [12, 1], [12, 12]] as const;

function stampPlayer(fr: Uint8Array, F: number, s: CraftaxState): void {
  const needs = [s.drink <= 3, s.food <= 3, s.energy <= 3, roundEven(s.health) <= 3];
  needs.forEach((on, k) => { if (on) for (const [y, x] of HOLES[k]!) fr[(TILE_Y + y) * F + TILE_X + x] = 0; });
  for (const [y0, x0] of CORNERS.slice(0, Math.min(4, Math.max(0, s.inventory.pickaxe))))
    for (let y = y0; y < y0 + 3; y++) for (let x = x0; x < x0 + 3; x++) fr[(TILE_Y + y) * F + TILE_X + x] = 1;
}

interface ViewTiles {
  /** each tile's glyph */
  tile: Int32Array;
  /** each tile's health bar (0: none) */
  hp: Float32Array;
  lit: Uint8Array;
  /** the creature class on each tile, -1 none */
  entCls: Int8Array;
}

function viewTiles(A: CraftaxAtlas, s: CraftaxState): ViewTiles {
  const [H, W] = A.view, NB = A.n_blocks, MS = s.size[0]!;
  const cy = H >> 1, cx = W >> 1, py = s.pos[0]!, px = s.pos[1]!, oob = A.blocks.indexOf("out of bounds");
  const tile = new Int32Array(H * W), hp = new Float32Array(H * W), lit = new Uint8Array(H * W), entCls = new Int8Array(H * W).fill(-1);
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) {
      const y = py - cy + r, x = px - cx + c, k = r * W + c, inside = y >= 0 && y < MS && x >= 0 && x < MS;
      const blk = inside ? s.map[y * MS + x]! : oob, item = inside ? s.items[y * MS + x]! : 0;
      tile[k] = item > 0 ? NB + item - 1 : blk;
      lit[k] = inside && s.light[y * MS + x]! > VISIBLE ? 1 : 0;
    }
  const ent = new Int32Array(H * W).fill(-1);
  A.classes.forEach((cls, ci) => {
    for (const m of (s[CLASS_FIELD[cls]!] as CraftaxMob[] | undefined) ?? []) {
      const r = m.pos[0]! - py + cy, c = m.pos[1]! - px + cx;
      if (r < 0 || r >= H || c < 0 || c >= W) continue;
      let g = A.cls_base[ci]![m.type]!;
      if (m.dir) { const dy = m.dir[0]!, dx = m.dir[1]!; g += dx < 0 ? 0 : dx > 0 ? 1 : dy < 0 ? 2 : 3; }
      ent[r * W + c] = g; entCls[r * W + c] = ci; hp[r * W + c] = cls === "melee" || cls === "ranged" ? m.hp : 0;
    }
  });
  for (let k = 0; k < H * W; k++) {
    if (ent[k]! >= 0) tile[k] = ent[k]!;
    if (!lit[k]) tile[k] = A.dark;
    if (!(lit[k] && ent[k]! >= 0)) hp[k] = 0;
  }
  const ck = cy * W + cx;
  hp[ck] = 0; tile[ck] = A.player0 + Math.min(3, Math.max(0, s.dir - 1));
  return { tile, hp, lit, entCls };
}

export interface FrameOptions {
  /** the status row's mode pad, 0 … 3 (the game page draws 0) */
  mode?: number;
  /** mark the player's tile with its needs and pickaxe, as the crystals were trained (default true) */
  stamp?: boolean;
  atlas?: CraftaxAtlas;
}

/** The board picture for a game state: F × F bytes, 0 dark mirror, 1 lit. Memory cells blank. */
export function frame(s: CraftaxState, o: FrameOptions = {}): Uint8Array {
  const A = o.atlas ?? craftaxAtlas(), F = A.frame, TL = A.tile, [H, W] = A.view;
  const { tile, hp } = viewTiles(A, s);
  const fr = A.staticBits.slice(), G = A.glyphBits;
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) {
      const k = r * W + c, g0 = tile[k]! * TL * TL, bar = hp[k]! > 0, n = Math.min(TL, Math.max(0, Math.ceil(hp[k]! / 2)));
      for (let y = 0; y < TL; y++)
        for (let x = 0; x < TL; x++) {
          let b = G[g0 + y * TL + x]!;
          if (bar && y >= TL - 2) b = y === TL - 1 && x < n ? 1 : 0;
          fr[(r * TL + y) * F + c * TL + x] = b;
        }
    }
  hudOn(A, s, o.mode ?? 0).forEach((on, j) => on.forEach((lit, k) => { if (lit) for (const p of A.pads[j]![k]!) fr[p] = 1; }));
  if (o.stamp ?? true) stampPlayer(fr, F, s);
  return fr;
}

// A coarse view for simple walkers: each of the 9 × 11 tiles as 0 unseen, 1 open (you can walk on it), 2 solid, 3 useful, 4 water,
// 5 danger, from the same glyphs the picture draws; the player's tile takes the class of the item, else the block, under it.
const BLOCK_COARSE: Record<string, number> = {
  invalid: 0, "out of bounds": 0, darkness: 0, grass: 1, sand: 1, path: 1, gravel: 1, "fire grass": 1, "ice grass": 1,
  stone: 2, tree: 2, wood: 2, wall: 2, "mossy wall": 2, stalagmite: 2, "fire tree": 2, "ice shrub": 2, grave: 2, "grave 2": 2, "grave 3": 2,
  coal: 3, iron: 3, diamond: 3, sapphire: 3, ruby: 3, "crafting table": 3, furnace: 3, plant: 3, "ripe plant": 3, chest: 3, fountain: 3,
  "fire ench. table": 3, "ice ench. table": 3, water: 4, lava: 5, necromancer: 5, "necromancer (vulnerable)": 5,
};
const ITEM_COARSE: Record<string, number> = { torch: 1, "ladder down": 3, "ladder up": 3, "ladder down (blocked)": 3 };
const CLASS_COARSE: Record<string, number> = { passive: 3, melee: 5, ranged: 5, "mob projectile": 5, "your projectile": 1 };

export function viewCoarse(s: CraftaxState, atlas?: CraftaxAtlas): Uint8Array {
  const A = atlas ?? craftaxAtlas(), [H, W] = A.view, NB = A.n_blocks, MS = s.size[0]!, { tile, lit, entCls } = viewTiles(A, s);
  const kind = (g: number): number => (g < NB ? BLOCK_COARSE[A.blocks[g]!] : ITEM_COARSE[A.items[g - NB]!]) ?? 0;
  const out = new Uint8Array(H * W), ck = (H >> 1) * W + (W >> 1);
  for (let k = 0; k < H * W; k++) out[k] = !lit[k] ? 0 : entCls[k]! >= 0 ? CLASS_COARSE[A.classes[entCls[k]!]!]! : kind(tile[k]!);
  const at = s.pos[0]! * MS + s.pos[1]!, item = s.items[at]!;
  out[ck] = item > 0 ? kind(NB + item - 1) : kind(s.map[at]!);
  return out;
}
