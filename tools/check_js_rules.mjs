// Differential test: the browser engine (web/portable/senet-engine.js) against the Rust
// move-generation dump (docs/FORMATS.md), which must hold at least one record, each of a
// game in progress and a throw. Usage: node tools/check_js_rules.mjs runs/moves_1m.jsonl
import { createReadStream } from "node:fs";
import { createRequire } from "node:module";
import readline from "node:readline";

const require = createRequire(import.meta.url);
const E = require("../web/portable/senet-engine.js");

// The fields of a dumped move, which are compared in this order (a move's own key order
// does not matter; the order of the moves and of their squares does).
const MOVE_KEYS = ["from", "to", "kind", "dir", "me", "opp"];
const moveText = (m) => JSON.stringify(MOVE_KEYS.map((k) => m[k]));
const isMove = (m) => typeof m === "object" && m !== null && !Array.isArray(m)
  && Object.keys(m).length === MOVE_KEYS.length && MOVE_KEYS.every((k) => Object.hasOwn(m, k));

// A record's problem, or null. Its moves' values are checked by the comparison.
function recordProblem(r) {
  if (typeof r !== "object" || r === null || Array.isArray(r)) return "not a JSON object";
  const isSquares = (v) => Array.isArray(v) && v.length <= 5 && new Set(v).size === v.length
    && v.every((s) => Number.isInteger(s) && s >= 1 && s <= 30 && s !== 27);
  if (!isSquares(r.me) || !isSquares(r.opp) || r.me.some((s) => r.opp.includes(s))) return "not a valid position";
  if (!r.me.length || !r.opp.length) return "a finished game";
  if (!Number.isInteger(r.t) || r.t < 1 || r.t > 5) return "not a throw 1..5";
  if (!Array.isArray(r.moves)) return "no list of moves";
  if (!r.moves.every(isMove)) return `a move without exactly the keys ${MOVE_KEYS.join(", ")}`;
  return null;
}

const file = process.argv[2];
if (!file) {
  console.error("usage: node tools/check_js_rules.mjs <moves.jsonl>");
  process.exit(2);
}
const rl = readline.createInterface({ input: createReadStream(file), crlfDelay: Infinity });
let n = 0, bad = 0;
const report = (...what) => { if (++bad <= 5) console.log(...what); };
for await (const line of rl) {
  if (!line.trim()) continue;
  n++;
  let r, problem;
  try {
    r = JSON.parse(line);
    problem = recordProblem(r);
  } catch (e) {
    problem = e.message;
  }
  if (problem) {
    report(`MALFORMED record ${n}: ${problem}`);
    continue;
  }
  const got = E.legalMoves(E.maskOf(r.me), E.maskOf(r.opp), r.t).map((m) => ({
    from: m.from, to: m.to, kind: m.kind, dir: m.back ? "back" : "fwd", me: E.squares(m.me), opp: E.squares(m.opp),
  }));
  if (JSON.stringify(got.map(moveText)) !== JSON.stringify(r.moves.map(moveText))) {
    const position = JSON.stringify({ me: r.me, opp: r.opp, t: r.t });
    report("MISMATCH", position, "\n  rust:", JSON.stringify(r.moves), "\n  js:  ", JSON.stringify(got));
  }
}
console.log(`${n} positions checked, ${bad} mismatched or malformed`);
process.exit(bad || !n ? 1 : 0);
