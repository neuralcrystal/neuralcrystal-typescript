import { describe, expect, it } from "vitest";
import { huggingFaceUrl } from "../src/index.js";

describe("huggingFaceUrl", () => {
  it("names the published files on Hugging Face", () => {
    expect(huggingFaceUrl("cifar10")).toBe("https://huggingface.co/neuralcrystal/cifar10-crystal/resolve/main/cifar10-crystal-16bit.safetensors");
    expect(huggingFaceUrl("chess", 8)).toBe("https://huggingface.co/neuralcrystal/chess-crystal/resolve/main/chess-crystal-8bit.safetensors");
    expect(huggingFaceUrl("mnist", 32)).toBe("https://huggingface.co/neuralcrystal/mnist-crystal/resolve/main/mnist-crystal.safetensors");
    expect(huggingFaceUrl("craftax")).toBe("https://huggingface.co/neuralcrystal/craftax-crystal/resolve/main/craftax-crystal-16bit.nc16.gz");
    expect(() => huggingFaceUrl("craftax", 8)).toThrow();
  });
});
