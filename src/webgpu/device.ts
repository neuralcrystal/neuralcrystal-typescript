// Getting a WebGPU device for the light path, with the adapter's own limits (the column kernel wants N floats ×2 of workgroup memory).

export interface GpuInfo {
  device: GPUDevice;
  limits: GPUSupportedLimits;
  info: GPUAdapterInfo | null;
}

/** A device, or null when this browser has no usable WebGPU. `gpu` defaults to navigator.gpu. */
export async function requestGpu(gpu: GPU | undefined = globalThis.navigator?.gpu): Promise<GpuInfo | null> {
  if (!gpu) return null;
  const ad = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!ad) return null;
  const L = ad.limits;
  const device = await ad.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    },
  });
  return { device, limits: L, info: ad.info ?? null };
}

/** Can this device run an N×N grid? (the row/column kernels hold one line of N complex f32 in workgroup memory) */
export function gridFits(limits: GPUSupportedLimits, N: number): boolean {
  return N >= 64 && N <= 2048 && (N & (N - 1)) === 0 && 8 * N <= limits.maxComputeWorkgroupStorageSize && 8 * N * N <= limits.maxStorageBufferBindingSize;
}
