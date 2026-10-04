// Network parity: the browser engine's forward pass (web/portable/senet-engine.js) against
// reference outputs, compared as logits (the reference's from its probability, in double
// precision). Weights are passed base64-encoded, exactly as the portable build does.
// Usage: node tools/check_js_net.mjs models/senet_net.bin cases.json
// cases.json: [[me_mask, opp_mask, expected_probability], ...], at least one case, each a
// game in progress.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { readSnn1 } from "./snn1.mjs";

const require = createRequire(import.meta.url);
const E = require("../web/portable/senet-engine.js");

const TOL = 5e-6; // as NET_TOL in python/senet_train/check_net.py
const logit = (p) => Math.log(p) - Math.log1p(-p);

// A case's problem, or null: masks of a game in progress and a probability.
const USABLE = 0x7fff_fffe & ~(1 << 27); // squares 1..30 except 27
function caseProblem(c) {
  if (!Array.isArray(c) || c.length !== 3) return "not [me_mask, opp_mask, probability]";
  const [me, opp, want] = c;
  const isMask = (m) => Number.isInteger(m) && m > 0 && (m & ~USABLE) === 0;
  if (!isMask(me) || !isMask(opp) || (me & opp) !== 0) return "not the masks of a game in progress";
  if (typeof want !== "number" || !(want >= 0 && want <= 1)) return "the expected value is not a probability";
  return null;
}

const [binPath, casesPath] = process.argv.slice(2);
if (!binPath || !casesPath) {
  console.error("usage: node tools/check_js_net.mjs <net.bin> <cases.json>");
  process.exit(2);
}
const net = E.loadNet(readSnn1(binPath));
const cases = JSON.parse(readFileSync(casesPath, "utf8"));
if (!Array.isArray(cases) || !cases.length) {
  console.log(`${casesPath}: no cases`);
  process.exit(1);
}
let worst = 0;
for (const [i, c] of cases.entries()) {
  const problem = caseProblem(c);
  if (problem) {
    console.log(`${casesPath}: case ${i} ${JSON.stringify(c)}: ${problem}`);
    process.exit(1);
  }
  const [me, opp, want] = c;
  worst = Math.max(worst, Math.abs(E.netLogit(net, me, opp) - logit(want))); // NaN stays NaN
}
console.log(`js network: ${cases.length} positions, max |js - reference| logit = ${worst.toExponential(2)}`);
process.exit(worst < TOL ? 0 : 1);
