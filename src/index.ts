// The public surface of neuralcrystal: crystal files, the engine and its light backends, the hardware a page draws with, and the
// problem kits. Everything else is internal and may change between versions.

// crystal files
export type { CrystalData, ModelSpec, PlaneSpec, QuadSpec, QuadSplitSpec, ImagerSpec, PhaseMap } from "./crystal/types.js";
export { loadCrystal, loadBytes, gunzip, isGzip } from "./crystal/load.js";
export { huggingFaceUrl, HUGGING_FACE } from "./crystal/hugging-face.js";
export type { PublishedCrystal } from "./crystal/hugging-face.js";
export type { LoadOptions, LoadProgress, CrystalSource } from "./crystal/load.js";
export { crystalLayout, readCrystal } from "./crystal/read.js";
export { packCrystal } from "./crystal/pack.js";
export type { PackOptions } from "./crystal/pack.js";
export { parseSafetensors, writeSafetensors, isSafetensors, tensorF32 } from "./crystal/safetensors.js";
export { mirrorCrystal, mirrorFrame } from "./crystal/mirror.js";

// the engine
export { CrystalEngine } from "./engine/crystal-engine.js";
export { EngineDisposed } from "./engine/types.js";
export type { RunResult, SplitResult, LightTrace, StageTrace, QuadTrace, PassHook } from "./engine/types.js";
export type { Geometry } from "./physics/geometry.js";
export { geometryOf, rnd } from "./physics/geometry.js";
export { mapWindow, spreadMap } from "./physics/phases.js";
export type { HandoffConfig } from "./hardware/handoff.js";

// light backends: the CPU reference, WebGPU, and the interface to write another
export type { Field } from "./light/field.js";
export { makeField, cloneField, intensity } from "./light/field.js";
export type { LightBackend, Crop } from "./light/backend.js";
export type { Op, SectionPlan } from "./physics/plan.js";
export { CpuBackend } from "./cpu/cpu-backend.js";
export { FFT } from "./cpu/fft.js";
export { WebGpuBackend } from "./webgpu/webgpu-backend.js";
export type { WebGpuOptions } from "./webgpu/webgpu-backend.js";
export { requestGpu, gridFits } from "./webgpu/device.js";
export type { GpuInfo } from "./webgpu/device.js";

// hardware: detectors, energy, the player's screen, handwriting
export type { DetectorLayout, Rect, ClassReading } from "./hardware/detectors.js";
export { DETECTOR_LAYOUTS, detectorLayout, detectorRects, detectorLight, readClasses } from "./hardware/detectors.js";
export type { PowerSpec, LightSpec } from "./hardware/energy.js";
export { DEMO_POWER, BENCH_POWER, SENSOR_E, SENSOR_QE, standbyW, awakeW, pulseLightJ, requestJ, pulseJ } from "./hardware/energy.js";
export type { SteadyInput } from "./hardware/battery.js";
export { Battery, BATT_AA_J, BATT_AAA_J, BATT_FRESH_PCT } from "./hardware/battery.js";
export { Panel, OLED_PX, OLED_RGB } from "./hardware/screen/panel.js";
export { glyph } from "./hardware/screen/face57.js";
export type { StrokePoint } from "./input/stroke-input.js";
export { StrokeInput } from "./input/stroke-input.js";

// problem kits
export type { ProblemKit, Solved } from "./kits/types.js";
export { solve, solveTraced } from "./kits/types.js";

export type { DigitShard } from "./problems/mnist/dataset.js";
export { parseShard, digitOf, findDigit } from "./problems/mnist/dataset.js";
export type { MnistKit, MnistAnswer, MnistInput } from "./problems/mnist/kit.js";
export { mnistKit, MNIST_PEN } from "./problems/mnist/kit.js";
export type { DigitScreenView } from "./problems/mnist/digit-screen.js";
export { panelToField, inDrawZone } from "./problems/mnist/digit-screen.js";

export type { CifarKit, CifarAnswer } from "./problems/cifar/kit.js";
export { cifarKit } from "./problems/cifar/kit.js";
export type { CifarShard, CifarPicture } from "./problems/cifar/dataset.js";
export { CIFAR_NAMES, parseCifarShard, cifarPicture, flipPicture } from "./problems/cifar/dataset.js";
export type { CifarScreenView } from "./problems/cifar/screen.js";

export type { CheckersKit, CheckersAnswer } from "./problems/checkers/kit.js";
export { checkersKit } from "./problems/checkers/kit.js";
export type { CheckersState, CheckersMove, CheckersStatus, MoveReading, ReadOptions } from "./problems/checkers/board.js";
export { checkers } from "./problems/checkers/board.js";
export type { CheckersScreenView } from "./problems/checkers/screen.js";
export { screenSquare, seatSquare } from "./problems/checkers/screen.js";
export type { ThinkOptions, ThinkNote } from "./problems/checkers/think.js";
export { think, repRule, repNext, valueCells, valueOf } from "./problems/checkers/think.js";
export type { CheckersGhost } from "./problems/checkers/ghost.js";
export { checkersGhost } from "./problems/checkers/ghost.js";

export type { ChessKit, ChessAnswer } from "./problems/chess/kit.js";
export { chessKit, chessValueCells, chessValueOf } from "./problems/chess/kit.js";
export type { ChessState, ChessMove, ChessStatus, ChessReading, ChessReadOptions } from "./problems/chess/board.js";
export { chess, mirror as chessMirror, mirrorMove as chessMirrorMove } from "./problems/chess/board.js";
export type { ChessScreenView } from "./problems/chess/screen.js";
export { chessScreenSquare } from "./problems/chess/screen.js";
export type { ChessThinkOptions, ChessThought, ChessRules } from "./problems/chess/think.js";
export { chessThink, thoughtReading, chessRules, repNext as chessRepNext, CHESS_RULES, orderedReading } from "./problems/chess/think.js";
export type { GhostPiece } from "./problems/chess/ghost.js";
export { ghostPieces } from "./problems/chess/ghost.js";

export type { CraftaxAtlas, CraftaxAtlasJson } from "./problems/craftax/atlas.js";
export { craftaxAtlas } from "./problems/craftax/atlas.js";
export type { CraftaxState, CraftaxMob, CraftaxInventory } from "./problems/craftax/state.js";
export { CRAFTAX_ACTIONS, CRAFTAX_N_ACTIONS, actionMask as craftaxActionMask } from "./problems/craftax/state.js";
export type { FrameOptions as CraftaxFrameOptions } from "./problems/craftax/frame.js";
export { frame as craftaxFrame, viewCoarse as craftaxViewCoarse } from "./problems/craftax/frame.js";
export type { CraftaxLayout, CraftaxCell } from "./problems/craftax/layout.js";
export { craftaxLayout, pictures as craftaxPictures } from "./problems/craftax/layout.js";
export type { CraftaxReading } from "./problems/craftax/read.js";
export { readCraftax } from "./problems/craftax/read.js";
export type { CraftaxEngine, CraftaxMove, AgentOptions as CraftaxAgentOptions } from "./problems/craftax/agent.js";
export { CraftaxAgent, craftaxStep, chooseMove as craftaxChooseMove } from "./problems/craftax/agent.js";
