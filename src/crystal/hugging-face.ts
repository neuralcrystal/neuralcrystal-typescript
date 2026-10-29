// Where the published crystals live: the neuralcrystal organization on Hugging Face. huggingFaceUrl("chess") is the file the browser loads.
export type PublishedCrystal = "mnist" | "cifar10" | "checkers" | "chess" | "craftax";

export const HUGGING_FACE = "https://huggingface.co/neuralcrystal";

/**
 * The URL of a published crystal: 16 bits a sample by default (what the browser loads), or 8, or 32 (the full fp32 file).
 * Craftax ships its 16-bit crystal in the compact web format (.nc16.gz) and has no 8-bit file.
 */
export function huggingFaceUrl(name: PublishedCrystal, bits: 8 | 16 | 32 = 16): string {
  const repo = `${name}-crystal`;
  if (name === "craftax" && bits === 8) throw new Error("the Craftax crystal has no 8-bit file: use 16 or 32");
  const file = bits === 32 ? `${repo}.safetensors` : name === "craftax" ? `${repo}-16bit.nc16.gz` : `${repo}-${bits}bit.safetensors`;
  return `${HUGGING_FACE}/${repo}/resolve/main/${file}`;
}
