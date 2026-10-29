// GPU buffers kept by key, least recently used destroyed first.

export class LruBuffers {
  private readonly map = new Map<string, GPUBuffer>();
  constructor(private readonly cap: number) {}

  get(key: string, make: () => GPUBuffer): GPUBuffer {
    let b = this.map.get(key);
    if (b) {
      this.map.delete(key);
      this.map.set(key, b);
      return b;
    }
    b = make();
    this.map.set(key, b);
    while (this.map.size > this.cap) {
      const [k, old] = this.map.entries().next().value as [string, GPUBuffer];
      old.destroy();
      this.map.delete(k);
    }
    return b;
  }

  /** bytes held */
  get bytes(): number {
    let n = 0;
    for (const b of this.map.values()) n += b.size;
    return n;
  }

  clear(): void {
    for (const b of this.map.values()) b.destroy();
    this.map.clear();
  }
}
