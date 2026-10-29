// The crystal's recipe: the v1 model object the simulation and the Python trainer write.
// Only the fields the engine reads are typed; everything else rides along untouched in `extra`-style index access.

/** One wafer of the stack. `kinds`/`params`/`arrs` give a role per quad section; without them every section uses `kind`. */
export interface PlaneSpec {
  kind: string;
  param?: number;
  dist_mm?: number;
  off?: boolean;
  seed?: number | null;
  hold?: string | null;
  n?: number;
  steps?: number;
  kinds?: string[];
  params?: number[];
  arrs?: (LensArraySpec | null)[];
  arr?: LensArraySpec | null;
}

export interface LensArraySpec {
  nx?: number;
  ny?: number;
  pitch_mm?: number;
  pitch?: number;
  cells?: (number | null)[];
}

export interface ImagerSpec {
  g?: number;
  b?: number;
  gm?: number;
  flip?: boolean;
  inv?: boolean;
  /** "ideal" | "bonded" | "mirror1" | "mirror2" | "cell8" | a relay factor k */
  sen?: string | number;
  mode?: string;
  style?: string;
}

export interface QuadNormSpec {
  t?: number;
  s?: number;
}

export interface QuadSpec {
  n?: number;
  mir?: number;
  levels?: number;
  leak?: number;
  gains?: number[];
  ghost?: number;
  res?: number;
  unrot?: boolean;
  band?: number;
  bandY0?: number;
  norm?: QuadNormSpec | null;
  hidden?: number;
  bnd?: unknown[] | null;
  passes?: number[] | null;
  /** SPLIT HEADS (the Python package's run_split): two input sections side by side, the per-pixel combine, the middle sections, the output sections */
  split?: QuadSplitSpec | null;
  tilePx?: number;
  gapUm?: number;
  slotUm?: number;
}

export interface QuadSplitSpec {
  /** the two input heads' sections (pass 1, each on its own frame) */
  in: number[];
  /** the middle sections, in a row */
  mid: number[];
  /** the output sections, side by side on the last pass's picture */
  out: number[];
  /** THE EARLIER SPLIT (the Python package's run_split): after the middle, each head its own row of sections ending in its out section; absent: both outs on the last middle picture */
  branch?: number[][] | null;
  /** the controller's per-pixel combine a·x + b·y + c (F × F each, row-major); absent: a = b = 0.75, c = 0 */
  combine?: { a?: number[] | null; b?: number[] | null; c?: number[] | null } | null;
  test?: string;
}

export interface OverlaySpec {
  on?: boolean;
  mode?: string;
}

export interface NonlinearSpec {
  abs?: boolean;
}

/** The model object: `__metadata__.neuralcrystal` of a v2 file, or a whole v1 JSON file. */
export interface ModelSpec {
  format?: string;
  N: number;
  dx_um: number;
  lam_um?: number;
  n_medium?: number;
  inGap_mm?: number;
  ext?: number;
  outScale?: number;
  problem?: string;
  expect?: string;
  featUm?: number;
  wholeGlass?: boolean;
  winUm?: number;
  solveMirror?: boolean;
  sideBlack?: boolean;
  absorbRim?: boolean;
  subHopUm?: number;
  zMaxSplit?: boolean;
  dmdNA?: number;
  dmdMirrors?: number;
  imager?: ImagerSpec;
  quad?: QuadSpec | null;
  overlay?: OverlaySpec | null;
  nl?: NonlinearSpec | null;
  planes: PlaneSpec[];
  [key: string]: unknown;
}

/** A learned map: row-major float32, `n × n`, radians (unwrapped, as trained). */
export type PhaseMap = Float32Array;

/** The recipe plus every learned map, per plane and per section. `maps[i][k]` is plane i's map in section k (null: none). */
export interface CrystalData {
  model: ModelSpec;
  maps: (PhaseMap | null)[][];
  /** each plane's map size n (maps are n × n), 0 when it has none */
  mapN: number[];
  /** mirrored left to right (mirrorCrystal): the kits mirror the DMD picture and the detector squares to match */
  mirrored?: boolean;
  /** the web form's steps (packCrystal): 8 — every map value is k·2π/256, so a phone's GPU can hold the step k alone */
  bits?: 8 | 16;
  /** the maps are Uint8Array 8-bit steps, not radians (LoadOptions.stepMaps): read them through spreadMap / mapWindow */
  steps?: boolean;
}
