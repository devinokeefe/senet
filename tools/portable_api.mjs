// The portable build's in-browser API (web/portable/senet-engine.js + local-api.js), loaded
// in node:vm as the page loads it: the scripts share one global scope, with a small
// deterministic network (base64 weights, as the portable build embeds them).
// Used by tools/check_local_api.mjs and tools/check_web_app.mjs.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const SCRIPTS = ["web/portable/senet-engine.js", "web/portable/local-api.js"];

function fakeNet() {
  let i = 0;
  const weight = () => Math.sin(++i * 12.9898) * 0.2; // deterministic, roughly uniform
  const b64 = (n) => Buffer.from(Float32Array.from({ length: n }, weight).buffer).toString("base64");
  const layer = (nIn, nOut) => ({ in: nIn, out: nOut, w: b64(nIn * nOut), b: b64(nOut) });
  return { format: "senet-mlp-v1", layers: [layer(72, 16), layer(16, 1)] };
}

/** The page's `window.SENET_LOCAL_API(path, body)`. */
export function loadLocalApi() {
  const page = vm.createContext({ atob, SENET_NET: fakeNet() });
  page.window = page;
  for (const f of SCRIPTS) {
    vm.runInContext(readFileSync(new URL(`../${f}`, import.meta.url), "utf8"), page, { filename: f });
  }
  return page.SENET_LOCAL_API;
}
