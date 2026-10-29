// A Craftax game state, as the browser game engine's state JSON gives it (the Python package's craftax.state_dict makes the same
// object from a JAX state), the names of the 43 moves, and the moves the game allows.

/** the 43 moves, in Craftax's order */
export const CRAFTAX_ACTIONS = [
  "noop", "left", "right", "up", "down", "do", "sleep", "place stone", "place table", "place furnace", "place plant",
  "make wood pickaxe", "make stone pickaxe", "make iron pickaxe", "make wood sword", "make stone sword", "make iron sword", "rest",
  "descend", "ascend", "make diamond pickaxe", "make diamond sword", "make iron armor", "make diamond armor", "shoot arrow",
  "make arrow", "cast fireball", "cast iceball", "place torch", "drink red potion", "drink green potion", "drink blue potion",
  "drink pink potion", "drink cyan potion", "drink yellow potion", "read book", "enchant sword", "enchant armor", "make torch",
  "level up dexterity", "level up strength", "level up intelligence", "enchant bow",
] as const;
export const CRAFTAX_N_ACTIONS = 43;

/** a creature or projectile on the current floor */
export interface CraftaxMob {
  /** its type within its class */
  type: number;
  /** [row, col] on the map */
  pos: [number, number] | number[];
  hp: number;
  /** projectiles: the heading [dy, dx] */
  dir?: [number, number] | number[];
}

export interface CraftaxInventory {
  wood: number; stone: number; coal: number; iron: number; diamond: number; sapling: number;
  pickaxe: number; sword: number; bow: number; arrows: number; torches: number; ruby: number; sapphire: number; books: number;
  /** helmet, chestplate, pants, boots: 0 none, 1 iron, 2 diamond */
  armour: number[];
  /** red, green, blue, pink, cyan, yellow */
  potions: number[];
}

/** The fields the crystal's pictures and the move rules read (the engine's JSON carries more; extra fields are ignored). */
export interface CraftaxState {
  /** the floor, 0 (overworld) … 8 */
  level: number;
  /** the current floor's blocks, row-major, size[0] × size[1] */
  map: ArrayLike<number>;
  /** the current floor's items (0 none, 1 torch, 2 ladder down, 3 ladder up, 4 blocked ladder) */
  items: ArrayLike<number>;
  /** the light on each tile, 0 … 255 */
  light: ArrayLike<number>;
  size: [number, number] | number[];
  /** [row, col] */
  pos: [number, number] | number[];
  /** facing: 1 left, 2 right, 3 up, 4 down */
  dir: number;
  /** the overworld's daylight, 0 … 1 */
  daylight: number;
  health: number; food: number; drink: number; energy: number; mana: number;
  sleeping: boolean; resting: boolean;
  xp: number; dexterity: number; strength: number; intelligence: number;
  inventory: CraftaxInventory;
  sword_enchant: number; bow_enchant: number; armour_enchant: number[];
  /** fireball, iceball learned */
  spells: boolean[];
  boss_spawn_turns: number;
  /** kills on the current floor */
  monsters_killed: number;
  melee: CraftaxMob[]; ranged: CraftaxMob[]; passive: CraftaxMob[];
  mob_projectiles: CraftaxMob[]; player_projectiles: CraftaxMob[];
}

const LADDER_DOWN = 2, LADDER_UP = 3;

/**
 * The 43 moves the game allows (Craftax's own rules, as PufferLib's compute_action_mask and the Python package's craftax.action_mask):
 * moving and "do" always; crafting needs its materials; asleep or resting, only noop.
 */
export function actionMask(s: CraftaxState): boolean[] {
  const m = new Array<boolean>(CRAFTAX_N_ACTIONS).fill(false);
  m[0] = true;
  if (s.sleeping || s.resting) return m;
  const inv = s.inventory, { wood, stone, coal, iron, diamond: dia, pickaxe: pick, sword, arrows, torches } = inv, arm = inv.armour;
  const MS = s.size[0]!, item = s.items[s.pos[0]! * MS + s.pos[1]!] ?? 0, lv = s.level;
  const ench = s.mana >= 9 && (inv.ruby > 0 || inv.sapphire > 0);
  const allowed: [number, boolean][] = [
    [1, true], [2, true], [3, true], [4, true], [5, true],
    [6, s.energy < 7 + 2 * s.dexterity], [17, s.health < 8 + s.strength],
    [7, stone > 0], [9, stone > 0], [8, wood >= 2], [10, inv.sapling > 0], [28, torches > 0],
    [11, wood > 0 && pick < 1], [12, wood > 0 && stone > 0 && pick < 2], [13, wood > 0 && stone > 0 && iron > 0 && coal > 0 && pick < 3],
    [20, wood > 0 && dia >= 3 && pick < 4], [14, wood > 0 && sword < 1], [15, wood > 0 && stone > 0 && sword < 2],
    [16, wood > 0 && stone > 0 && iron > 0 && coal > 0 && sword < 3], [21, wood > 0 && dia >= 2 && sword < 4],
    [25, wood > 0 && stone > 0 && arrows < 99], [38, wood > 0 && coal > 0 && torches < 99],
    [22, arm.some((a) => a < 1) && iron >= 3 && coal >= 3], [23, arm.some((a) => a < 2) && dia >= 3],
    [18, item === LADDER_DOWN && s.monsters_killed >= 8 && lv < 8], [19, item === LADDER_UP && lv > 0],
    [24, inv.bow > 0 && arrows > 0], [26, !!s.spells[0] && s.mana >= 2], [27, !!s.spells[1] && s.mana >= 2],
    [35, inv.books > 0], [36, ench && sword > 0], [37, ench && arm.reduce((a, b) => a + b, 0) > 0], [42, ench && inv.bow > 0],
    [39, s.xp > 0 && s.dexterity < 5], [40, s.xp > 0 && s.strength < 5], [41, s.xp > 0 && s.intelligence < 5],
  ];
  for (const [k, v] of allowed) m[k] = v;
  for (let i = 0; i < 6; i++) m[29 + i] = (inv.potions[i] ?? 0) > 0;
  return m;
}
