"use strict";
// In-browser implementation of the server API used by app.js (crates/senet-cli/src/server.rs,
// docs/API.md) for the portable single-file build: the distilled neural network (+ optional
// 1-throw expectimax) replaces the solved database. Replies have the server's keys and errors
// its messages (for a malformed body, only its "bad request: " prefix, and for an unknown
// field the message without serde's position); /api/info adds `label` and `tagline` for the
// page header. The random bot is seeded by `seed` (default 0, as on the server) but does not
// reproduce the Rust RNG's choices. tools/check_local_api.mjs checks this against the server.
(function () {
  const E = window.SenetEngine;
  let NET = null;
  const net = () => (NET ??= E.loadNet(window.SENET_NET));
  const evalFn = (me, opp) => E.netValue(net(), me, opp);

  const PIECES = 5;
  const THROWS = [1, 2, 3, 4, 5];
  // The bots /api/ai accepts, strongest first. "perfect" is played by "net:1" (as the server
  // plays it without the database), so /api/info does not list it; the others search this
  // many throws past the move, except "random".
  const BOTS = ["perfect", "net:1", "net", "random"];
  const DEPTH = { "net:1": 1, "net": 0 };
  // Hints use the strongest bot's lookahead; the server uses its best evaluator directly.
  const HINT_DEPTH = DEPTH["net:1"];
  // Hints flag every move within this of the best value as best (as server.rs does).
  const BEST_TOLERANCE = 1e-9;
  const TIE_TOLERANCE = 1e-12;
  const other = (c) => (c === "white" ? "black" : "white");

  // ---- Request bodies, read as the server's serde structs read them. Messages about a
  // malformed body share only the server's "bad request: " prefix. ----
  const fail = (msg) => {
    throw new Error(msg);
  };
  const bad = (expected, v) => fail(`bad request: expected ${expected}, got ${JSON.stringify(v)}`);
  const unsigned = (bits) => (v) => (Number.isInteger(v) && v >= 0 && v < 2 ** bits ? v : bad(`u${bits}`, v));
  const u8 = unsigned(8), u32 = unsigned(32), u64 = unsigned(64);
  const list = (item) => (v) => (Array.isArray(v) ? v.map(item) : bad("a list", v));
  const string = (v) => (typeof v === "string" ? v : bad("a string", v));
  const optional = (read) => (v) => (v === null ? null : read(v));
  const variant = (...names) => (v) => (names.includes(v) ? v : bad(`one of ${names.join(", ")}`, v));

  // Each field: [reader, value when absent]; REQUIRED fields must be present.
  const REQUIRED = Symbol("required");
  const POSITION = { white: [list(u32), REQUIRED], black: [list(u32), REQUIRED], turn: [variant("white", "black"), REQUIRED] };
  const ANALYZE = { ...POSITION, throw: [optional(u8), null], eval: [variant("auto", "perfect", "net", "heuristic"), "auto"] };
  const AI = { ...POSITION, throw: [u8, REQUIRED], bot: [optional(string), null], seed: [u64, 0] };

  function parse(body, fields) {
    if (typeof body !== "object" || body === null || Array.isArray(body)) bad("a JSON object", body);
    const names = Object.keys(fields);
    for (const key of Object.keys(body)) {
      if (!names.includes(key)) {
        fail(`bad request: unknown field \`${key}\`, expected one of ${names.map((n) => `\`${n}\``).join(", ")}`);
      }
    }
    const req = {};
    for (const [key, [read, absent]] of Object.entries(fields)) {
      if (Object.hasOwn(body, key)) req[key] = read(body[key]);
      else if (absent === REQUIRED) fail(`bad request: missing field '${key}'`);
      else req[key] = absent;
    }
    return req;
  }

  // ---- Checks of the request's meaning, as Pos::from_squares and server.rs make them ----
  function mask(squares) {
    const seen = new Set();
    for (const s of squares) {
      if (s < 1 || s > 30 || s === 27) fail(`invalid square ${s} (squares are 1..=30, never 27)`);
      if (seen.has(s)) fail(`square ${s} listed twice`);
      seen.add(s);
    }
    return E.maskOf(squares);
  }
  // The position from the point of view of `turn`, which must be a game in progress.
  function moverView(req) {
    const mover = req.turn;
    const mine = req[mover], theirs = req[other(mover)];
    const me = mask(mine), opp = mask(theirs);
    if (mine.some((s) => theirs.includes(s))) fail("both sides occupy the same square");
    if (mine.length > PIECES || theirs.length > PIECES) fail(`at most ${PIECES} pieces per side`);
    if (me === 0 || opp === 0) fail("the game is already over");
    return { mover, me, opp };
  }
  const checkThrow = (t) => (t >= 1 && t <= 5 ? t : fail("throw must be 1..5"));

  // ---- Replies ----
  function moveJson(m, t, mover, winProb) {
    const done = m.me === 0;
    const [white, black] = mover === "white" ? [m.me, m.opp] : [m.opp, m.me];
    return {
      from: m.from, to: m.to, kind: m.kind, dir: m.back ? "back" : "fwd",
      white: E.squares(white), black: E.squares(black),
      next_turn: done ? null : E.EXTRA[t] ? mover : other(mover),
      winner: done ? mover : null,
      win_prob: winProb,
    };
  }

  // Value of each move for the mover, searching `depth` throws past the move. The last
  // result is kept: an AI turn asks for the same values twice (/api/analyze, then /api/ai
  // with net:1), and each search takes a while.
  let last = { key: null, values: null };
  function moveValues(v, t, moves, depth) {
    const key = `${v.me} ${v.opp} ${t} ${depth}`;
    if (last.key !== key) last = { key, values: moves.map((m) => E.childValue(evalFn, m, t, depth)) };
    return last.values;
  }

  // mulberry32: a small generator of uniform numbers in [0, 1), seeded with the low 32
  // bits of `seed`.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
    };
  }

  function info() {
    return {
      ruleset: "kendall5",
      database: null,
      network: true,
      bots: BOTS.filter((b) => b !== "perfect"),
      start: { white: [1, 3, 5, 7, 9], black: [2, 4, 6, 8, 10], turn: "white" },
      throw_probs: Object.fromEntries(THROWS.map((t) => [t, E.THROW_PROBS[t]])),
      extra_throws: THROWS.filter((t) => E.EXTRA[t]),
      label: "Portable AI: <b>distilled neural net</b>, running in your browser",
      tagline: "The game of passing — against a neural net distilled from perfect play",
    };
  }

  // `eval` is checked as on the server, but the network is the only evaluator here (the
  // reply's `source` says so).
  function analyze(body) {
    const req = parse(body, ANALYZE);
    const v = moverView(req);
    const p = evalFn(v.me, v.opp);
    const out = { source: "net", white_win_prob: v.mover === "white" ? p : 1 - p };
    if (req.throw !== null) {
      const t = checkThrow(req.throw);
      const moves = E.legalMoves(v.me, v.opp, t);
      const vals = moveValues(v, t, moves, HINT_DEPTH);
      const best = Math.max(...vals);
      out.throw = t;
      out.moves = moves.map((m, i) => ({ ...moveJson(m, t, v.mover, vals[i]), best: best - vals[i] < BEST_TOLERANCE }));
      if (!moves.length) out.pass_to = other(v.mover);
    }
    return out;
  }

  function ai(body) {
    const req = parse(body, AI);
    const v = moverView(req);
    const t = checkThrow(req.throw);
    const requested = req.bot ?? "perfect";
    if (!BOTS.includes(requested)) fail(`unknown bot '${requested}' (available: ${BOTS.join(", ")})`);
    const bot = requested === "perfect" ? "net:1" : requested;
    const moves = E.legalMoves(v.me, v.opp, t);
    if (!moves.length) return { choice: null, moves: [], bot };
    let choice, vals;
    if (bot === "random") {
      choice = Math.floor(mulberry32(req.seed)() * moves.length);
      vals = moves.map(() => null); // the server reports no values for the random bot
    } else {
      vals = moveValues(v, t, moves, DEPTH[bot]);
      // The first of near-equal moves, as the bots break ties (bots::TIE_TOLERANCE).
      choice = 0;
      for (let i = 1; i < vals.length; i++) if (vals[i] > vals[choice] + TIE_TOLERANCE) choice = i;
    }
    return { choice, moves: moves.map((m, i) => moveJson(m, t, v.mover, vals[i])), bot };
  }

  const ROUTES = { "/api/info": info, "/api/analyze": analyze, "/api/ai": ai };

  window.SENET_LOCAL_API = async function (path, body) {
    const route = ROUTES[path];
    if (!route) throw new Error("unknown endpoint " + path);
    return route(body);
  };
})();
