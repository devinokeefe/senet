// Contract test for the portable build's in-browser API (web/portable/local-api.js): its
// responses must have exactly the keys and types of the server's (crates/senet-cli/src/server.rs),
// its errors the server's messages, and its answers be consistent with the rules. It runs as
// the page loads it (tools/portable_api.mjs), with a small deterministic network.
// Usage: node tools/check_local_api.mjs [--server http://127.0.0.1:8080]
//   --server also runs the same checks against a running `senet serve`.
import { loadLocalApi } from "./portable_api.mjs";

// ---- The server's response shapes. "key?" = optional; [spec] = array of spec. ----
const int = Number.isInteger;
const num = (v) => typeof v === "number" && Number.isFinite(v);
const str = (v) => typeof v === "string";
const bool = (v) => typeof v === "boolean";
const nul = (v) => v === null;
const oneOf = (...xs) => (v) => xs.includes(v);
const either = (a, b) => (v) => problems(v, a).length === 0 || problems(v, b).length === 0;
const COLOR = oneOf("white", "black");
const MOVE = {
  from: int, to: int, kind: oneOf("move", "swap", "off", "water"), dir: oneOf("fwd", "back"),
  white: [int], black: [int], next_turn: either(COLOR, nul), winner: either(COLOR, nul), win_prob: num,
};
const SPEC = {
  info: {
    ruleset: str, database: either(nul, { complete: bool, path: str }), network: bool, bots: [str],
    start: { white: [int], black: [int], turn: COLOR },
    throw_probs: { 1: num, 2: num, 3: num, 4: num, 5: num }, extra_throws: [int],
  },
  analysis: { source: oneOf("perfect", "net", "heuristic"), white_win_prob: num },
  analysisWithThrow: {
    source: oneOf("perfect", "net", "heuristic"), white_win_prob: num,
    throw: int, moves: [{ ...MOVE, best: bool }], "pass_to?": COLOR,
  },
  ai: { choice: either(int, nul), moves: [{ ...MOVE, win_prob: either(num, nul) }], bot: str },
};

// Mismatches between a value and a spec, as "path: problem" strings.
function problems(v, spec, path = "") {
  if (typeof spec === "function") return spec(v) ? [] : [`${path || "value"}: unexpected ${typeof v === "number" ? v : JSON.stringify(v)}`];
  if (Array.isArray(spec)) {
    if (!Array.isArray(v)) return [`${path}: expected an array`];
    return v.flatMap((x, i) => problems(x, spec[0], `${path}[${i}]`));
  }
  if (typeof v !== "object" || v === null || Array.isArray(v)) return [`${path}: expected an object`];
  const out = [];
  const keys = new Set(Object.keys(v));
  for (const [k, s] of Object.entries(spec)) {
    const key = k.replace(/\?$/, "");
    if (keys.delete(key)) out.push(...problems(v[key], s, `${path}.${key}`));
    else if (key === k) out.push(`${path}.${key}: missing`);
  }
  for (const k of keys) out.push(`${path}.${k}: not in the server's response`);
  return out;
}

// ---- Test positions ----
const EXTRA_THROWS = [1, 4, 5]; // RULES.md
const other = (c) => (c === "white" ? "black" : "white");
const POSITIONS = [
  { name: "start", white: [1, 3, 5, 7, 9], black: [2, 4, 6, 8, 10], turn: "white" },
  { name: "forced pass unless 3", white: [28], black: [1], turn: "white" },
  { name: "drowning (15, 14 taken)", white: [15, 14, 9], black: [26, 3], turn: "black" },
  { name: "backward only (blockade)", white: [10], black: [11, 12, 13], turn: "white" },
  { name: "bear off to win", white: [30], black: [5], turn: "white" },
];
// Expected special cases: [position name, throw, check(analysis), description].
const EXPECT = [
  ["forced pass unless 3", 1, (r) => r.moves.length === 0 && r.pass_to === "black", "no move: pass to black"],
  ["drowning (15, 14 taken)", 1, (r) => r.moves.some((m) => m.kind === "water" && m.from === 26 && m.to === 13 && m.black.includes(13)), "26+1 drowns to 13"],
  ["backward only (blockade)", 3, (r) => r.moves.length > 0 && r.moves.every((m) => m.dir === "back"), "only backward moves"],
  ["bear off to win", 1, (r) => r.moves.length === 1 && r.moves[0].winner === "white" && r.moves[0].win_prob === 1, "winning bear-off"],
  ["start", 4, (r) => r.moves.every((m) => m.next_turn === "white"), "extra throw keeps the turn"],
  ["start", 2, (r) => r.moves.every((m) => m.next_turn === "black"), "throw 2 passes the turn"],
];

let checks = 0;
const failures = [];
function check(ok, what) {
  checks++;
  if (!ok) failures.push(what);
}
function shape(v, spec, what) {
  const p = problems(v, spec);
  check(p.length === 0, `${what}: ${p.slice(0, 3).join("; ")}`);
}
// The request must fail with a message starting with `message`.
async function rejects(promise, what, message) {
  let error = null;
  try { await promise; } catch (e) { error = e; }
  check(error?.message.startsWith(message), `${what}: expected an error "${message}…", got ${error ? `"${error.message}"` : "a reply"}`);
}

async function run(label, api, local) {
  const info = await api("/api/info");
  shape(info, SPEC.info, `${label} info`);
  check(JSON.stringify(info.extra_throws) === JSON.stringify(EXTRA_THROWS), `${label} info.extra_throws = ${JSON.stringify(info.extra_throws)}`);
  check(Math.abs(Object.values(info.throw_probs).reduce((a, b) => a + b, 0) - 1) < 1e-12, `${label} throw_probs sum to 1`);
  // The bots listed are those that play as themselves.
  const bots = info.bots;
  check(bots.every((b) => !b.startsWith("net") || info.network), `${label} info.bots: net bots need a network`);
  check(!bots.includes("perfect") || info.database?.complete, `${label} info.bots: perfect needs the complete database`);

  for (const p of POSITIONS) {
    const pos = { white: p.white, black: p.black, turn: p.turn };
    const what = `${label} ${p.name}`;
    const base = await api("/api/analyze", pos);
    shape(base, SPEC.analysis, `${what} analyze`);
    check(base.white_win_prob >= 0 && base.white_win_prob <= 1, `${what}: white_win_prob in [0, 1]`);
    for (let t = 1; t <= 5; t++) {
      const r = await api("/api/analyze", { ...pos, throw: t });
      const at = `${what} throw ${t}`;
      shape(r, SPEC.analysisWithThrow, `${at} analyze`);
      check(r.throw === t, `${at}: echoes the throw`);
      check(r.moves.length ? !("pass_to" in r) : r.pass_to === other(p.turn), `${at}: pass_to iff no move`);
      const best = Math.max(...r.moves.map((m) => m.win_prob));
      check(r.moves.every((m) => m.best === Math.abs(m.win_prob - best) < 1e-9), `${at}: best flags the top moves`);
      check(new Set(r.moves.map((m) => m.from)).size === r.moves.length, `${at}: one move per piece`);
      for (const m of r.moves) {
        const done = m[p.turn].length === 0;
        check(m.winner === (done ? p.turn : null), `${at} ${m.from}: winner`);
        const next = done ? null : EXTRA_THROWS.includes(t) ? p.turn : other(p.turn);
        check(m.next_turn === next, `${at} ${m.from}: next_turn`);
        check(m.win_prob >= 0 && m.win_prob <= 1, `${at} ${m.from}: win_prob in [0, 1]`);
      }
      for (const [name, tt, ok, desc] of EXPECT) if (name === p.name && tt === t) check(ok(r), `${at}: ${desc}`);

      for (const bot of bots) {
        const a = await api("/api/ai", { ...pos, throw: t, bot, seed: 7 });
        const ai = `${at} ai ${bot}`;
        shape(a, SPEC.ai, ai);
        const strip = (m) => JSON.stringify([m.from, m.to, m.kind, m.dir, m.white, m.black, m.next_turn, m.winner]);
        check(JSON.stringify(a.moves.map(strip)) === JSON.stringify(r.moves.map(strip)), `${ai}: same moves as analyze`);
        check(r.moves.length ? int(a.choice) && a.choice < a.moves.length : a.choice === null, `${ai}: choice`);
        check(a.bot === bot, `${ai}: reported bot`);
        const vals = a.moves.map((m) => m.win_prob);
        // Locally, hints search as net:1 does.
        if (local && bot === "net:1") check(JSON.stringify(vals) === JSON.stringify(r.moves.map((m) => m.win_prob)), `${ai}: the values of the hints`);
        if (bot === "random") check(vals.every(nul), `${ai}: no values`);
        else if (vals.length) check(vals[a.choice] >= Math.max(...vals) - 1e-12, `${ai}: picks its best value`);
        if (bot === "random") {
          const again = await api("/api/ai", { ...pos, throw: t, bot, seed: 7 });
          check(again.choice === a.choice, `${ai}: the same seed gives the same choice`);
        }
      }
    }
  }

  // The random bot's seeds reach every move (start position, throw 1: five moves).
  const start = { white: [1, 3, 5, 7, 9], black: [2, 4, 6, 8, 10], turn: "white" };
  // "perfect" is always accepted: unlisted, a stand-in plays it.
  const perfect = await api("/api/ai", { ...start, throw: 2, bot: "perfect" });
  shape(perfect, SPEC.ai, `${label} ai perfect`);
  check((perfect.bot === "perfect") === bots.includes("perfect"), `${label} ai perfect: played by ${perfect.bot}`);
  if (local) check(perfect.bot === "net:1", `${label} ai perfect: played by net:1`);
  const picks = new Set();
  for (let seed = 0; seed < 64; seed++) picks.add((await api("/api/ai", { ...start, throw: 1, bot: "random", seed })).choice);
  check(picks.size === 5, `${label} random bot: 64 seeds pick ${picks.size} of 5 moves`);

  // Rejected requests, with the start of the server's message: all of it for a request
  // whose body is well formed, the "bad request: " prefix for one whose body is not.
  const REJECTED = [
    ["square 27", "/api/analyze", { ...start, white: [27, 3, 5, 7, 9] }, "invalid square 27 (squares are 1..=30, never 27)"],
    ["square twice", "/api/analyze", { ...start, white: [3, 3, 5] }, "square 3 listed twice"],
    ["shared square", "/api/analyze", { ...start, white: [2, 3, 5, 7, 9] }, "both sides occupy the same square"],
    ["six pieces", "/api/analyze", { ...start, white: [1, 3, 5, 7, 9, 11] }, "at most 5 pieces per side"],
    ["finished game", "/api/analyze", { ...start, white: [] }, "the game is already over"],
    ["bad turn", "/api/analyze", { ...start, turn: "red" }, "bad request: "],
    ["no white", "/api/analyze", { black: start.black, turn: "white" }, "bad request: "],
    ["square -1", "/api/analyze", { ...start, white: [-1] }, "bad request: "],
    ["throw 6", "/api/analyze", { ...start, throw: 6 }, "throw must be 1..5"],
    ["throw 2.5", "/api/analyze", { ...start, throw: 2.5 }, "bad request: "],
    ["unknown eval", "/api/analyze", { ...start, eval: "oracle" }, "bad request: "],
    ["unknown field", "/api/analyze", { ...start, thorw: 2 }, "bad request: unknown field `thorw`, expected one of `white`, `black`, `turn`, `throw`, `eval`"],
    ["ai unknown field", "/api/ai", { ...start, throw: 2, bot: "random", sed: 1 }, "bad request: unknown field `sed`, expected one of `white`, `black`, `turn`, `throw`, `bot`, `seed`"],
    ["ai without throw", "/api/ai", { ...start, bot: "random" }, "bad request: "],
    ["ai throw 0", "/api/ai", { ...start, throw: 0, bot: "random" }, "throw must be 1..5"],
    ["ai seed -1", "/api/ai", { ...start, throw: 2, bot: "random", seed: -1 }, "bad request: "],
    ["unknown bot", "/api/ai", { ...start, throw: 2, bot: "nonsense" }, "unknown bot 'nonsense' (available: "],
    ["bot not offered", "/api/ai", { ...start, throw: 2, bot: "expectimax:4" }, "unknown bot 'expectimax:4' (available: "],
    ["unknown endpoint", "/api/nope", {}, "unknown endpoint /api/nope"],
  ];
  for (const [what, path, body, message] of REJECTED) await rejects(api(path, body), `${label} ${what}`, message);
}

function serverApi(base) {
  return async (path, body) => {
    const post = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(base + path, body ? post : {});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    return j;
  };
}

const flag = process.argv.indexOf("--server");
const server = flag < 0 ? null : (process.argv[flag + 1] ?? "http://127.0.0.1:8080").replace(/\/$/, "");
for (const [label, api, local] of [["local", loadLocalApi(), true], ...(server ? [["server", serverApi(server), false]] : [])]) {
  try {
    await run(label, api, local);
  } catch (e) {
    check(false, `${label}: crashed: ${e.message}`);
  }
}
for (const f of failures.slice(0, 10)) console.log("FAIL", f);
console.log(`local api: ${checks} checks, ${failures.length} failed${server ? ` (local and ${server})` : ""}`);
process.exit(failures.length ? 1 : 0);
