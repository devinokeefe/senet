// Reads an SNN1 network file into the portable build's form (tools/build_portable.py).
// Used by tools/check_js_net.mjs and tools/check_portable.mjs.
import { readFileSync } from "node:fs";

/**
 * Splits an SNN1 file (little-endian: b"SNN1", u32 layer count, then per layer u32 in,
 * u32 out, f32 weights[out][in], f32 bias[out]) into {format, layers: [{in, out, w, b}]}
 * with base64 weights, checking that every byte is used; the engine's loadNet checks the
 * rest.
 */
export function readSnn1(path) {
  const buf = readFileSync(path);
  let off = 0;
  const take = (n, what) => {
    if (off + n > buf.length) throw new Error(`${path}: truncated (${what}: ${n} bytes at offset ${off}, file has ${buf.length})`);
    off += n;
    return buf.subarray(off - n, off);
  };
  const u32 = (what) => take(4, what).readUInt32LE(0);
  if (take(4, "magic").toString("latin1") !== "SNN1") throw new Error(`${path}: not an SNN1 file`);
  const layers = [];
  for (let k = u32("layer count"); k > 0; k--) {
    const nIn = u32("layer inputs"), nOut = u32("layer outputs");
    const w = take(4 * nIn * nOut, "weights").toString("base64");
    const b = take(4 * nOut, "biases").toString("base64");
    layers.push({ in: nIn, out: nOut, w, b });
  }
  if (off !== buf.length) throw new Error(`${path}: ${buf.length - off} trailing bytes after the last layer`);
  return { format: "senet-mlp-v1", layers };
}
