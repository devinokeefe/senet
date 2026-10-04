// Smoke test of the web UI (web/app.js), run in node:vm on web/index.html with the DOM stub
// and fake timers of tools/page_stub.mjs. fetch is answered by the portable build's
// in-browser API (tools/portable_api.mjs); the test can hold requests back, make them fail
// or edit the replies. It plays AI-vs-AI games to the end, without asking for the moves'
// win chances that such a game never shows, and checks that undo, new game and a player
// change abort the requests in flight and drop replies that come anyway, that a forced move
// played by hand or no longer auto-played is not played for the human, that failed and
// timed-out requests (also with a plain-text error reply) on the human's and the AI's turn
// recover, that only the latest win chance is shown, that the accuracy report names the
// judge of each decision, that Space throws, and that an offline engine disables the
// controls.
// Usage: node tools/check_web_app.mjs
import { readFileSync } from "node:fs";
import { click, choosePlayer, loadPage, newGame, seeded, settle } from "./page_stub.mjs";
import { loadLocalApi } from "./portable_api.mjs";

const source = (f) => readFileSync(new URL(`../${f}`, import.meta.url), "utf8");
const other = (c) => (c === "white" ? "black" : "white");
const cap = (c) => c[0].toUpperCase() + c.slice(1);

// ---- The page: app.js with a controllable API ----
const STICKS = { 1: 0b0001, 2: 0b0011, 3: 0b0111, 4: 0b1111, 5: 0b0000 }; // light sides up

async function openPage({ offline = false } = {}) {
  const random = seeded(2026);
  const api = loadLocalApi();
  // hold(path, request): keep matching requests pending until release() (or until they are
  // aborted, unless `late`: then their replies are already on the way, and come anyway);
  // `aborted`: the paths of the requests aborted while held; fault: the next matching
  // request fails (also one released from hold), with a network error or an HTTP error
  // reply (`plain`: [status, statusText, text] of a reply that is not JSON);
  // edit(path, request, payload): the reply's payload instead; `requests`: [path, request]
  // of every request made.
  const net = {
    hold: null,
    held: [],
    late: false,
    aborted: [],
    fault: null,
    edit: null,
    requests: [],
    release() { for (const resume of net.held.splice(0)) resume(); },
  };
  const throws = []; // stick throws to deal before random ones

  const reply = (status, payload, statusText = "", text = JSON.stringify(payload)) => ({
    ok: status === 200, status, statusText,
    text: async () => text,
  });
  async function fetch(path, { method = "GET", headers = {}, body, signal } = {}) {
    if (!(signal instanceof AbortSignal)) throw new TypeError("a request that cannot be aborted");
    const request = body === undefined ? undefined : JSON.parse(body);
    if (body !== undefined && headers["Content-Type"] !== "application/json") {
      return reply(415, { error: "a request body must be application/json" });
    }
    net.requests.push([path, request]);
    if (offline) throw new TypeError("Failed to fetch");
    if (net.hold?.(path, request)) {
      await new Promise((resume, reject) => {
        net.held.push(resume);
        if (net.late) return;
        signal.addEventListener("abort", () => {
          net.aborted.push(path);
          reject(new DOMException("This operation was aborted", "AbortError"));
        });
      });
    }
    if (net.fault?.when(path, request)) {
      const { network, plain } = net.fault;
      net.fault = null;
      if (network) throw new TypeError("Failed to fetch");
      if (plain) return reply(plain[0], null, plain[1], plain[2]);
      return reply(500, { error: "injected failure" });
    }
    const expected = path === "/api/info" ? "GET" : "POST";
    if (method !== expected) return reply(405, { error: `use ${expected} for ${path}` });
    let payload;
    try {
      payload = await api(path, request);
    } catch (e) {
      return reply(400, { error: e.message });
    }
    return reply(200, net.edit ? net.edit(path, request, payload) : payload);
  }

  const crypto = {
    getRandomValues(buf) {
      buf[0] = throws.length ? STICKS[throws.shift()] : Math.floor(random() * 256);
      return buf;
    },
  };
  const page = await loadPage(source("web/index.html"), [["web/app.js", source("web/app.js")]], { random, fetch, crypto });
  return { ...page, net, throws, S: page.evaluate("S") }; // S: the game state, a top-level const of app.js
}

// ---- User actions ----
function pressSpace(page, target = page.document.body) {
  return page.document.fire("keydown", { key: " ", code: "Space", repeat: false, target });
}
const square = (page, s) => page.$("board").children.find((el) => Number(el.dataset.sq) === s);
const message = (page) => page.$("message").textContent;
const throwRequest = (path, request) => path === "/api/analyze" && request.throw !== undefined;
const probRequest = (path, request) => path === "/api/analyze" && request.throw === undefined;
const aiRequest = (path) => path === "/api/ai";
const probLabel = (page) => page.$("prob-white-label").textContent;

// ---- Checks ----
let checks = 0, games = 0, entries = 0;
const failures = [];
function check(ok, what) {
  checks++;
  if (!ok) failures.push(what);
}
process.on("unhandledRejection", (e) => check(false, `unhandled rejection: ${e?.stack ?? e}`));

async function scenario(name, run, options) {
  try {
    await run(await openPage(options));
  } catch (e) {
    check(false, `${name}: ${e?.stack ?? e}`);
  }
}

await scenario("AI vs AI", async (page) => {
  const { S, net } = page;
  for (const [white, black] of [["net:1", "random"], ["random", "net"], ["net", "net:1"], ["random", "random"]]) {
    const what = `${white} vs ${black}`;
    await newGame(page, white, black);
    net.requests.length = 0;
    const ended = await page.runUntil(() => S.winner !== null);
    check(!net.requests.some(([path, request]) => throwRequest(path, request)),
      `${what}: no throw is analyzed, as no move's win chance is shown`);
    check(net.requests.some(([path, request]) => probRequest(path, request)), `${what}: the win chance is asked for`);
    check(ended, `${what}: the game ends`);
    if (!ended) continue;
    games++;
    entries += S.log.length;
    check(S[S.winner].length === 0 && S[other(S.winner)].length > 0, `${what}: the winner has borne off every piece`);
    check(S.phase === "over" && S.history.length === 0, `${what}: the game is over and AI moves are not undoable`);
    check(page.$("turn-text").textContent === `${cap(S.winner)} wins!`, `${what}: the turn card names the winner`);
    check(page.$("log").children.length === S.log.length, `${what}: the record shows every entry`);
    check(!message(page).includes("failed"), `${what}: no request failed`);
  }
});

await scenario("Space", async (page) => {
  const { S } = page;
  await newGame(page, "human", "human");
  page.throws.push(3);
  let e = pressSpace(page, page.$("sel-white"));
  check(!e.defaultPrevented && S.phase === "throw", "Space on a select does not throw");
  e = pressSpace(page, page.$("btn-new"));
  check(!e.defaultPrevented && S.phase === "throw", "Space on a button reached by keyboard does not throw");
  page.document.fire("pointerdown");
  e = pressSpace(page, page.$("btn-new"));
  await settle();
  check(e.defaultPrevented && S.phase === "move" && S.throw === 3, "Space throws, also on a button that kept focus after a click");
  e = pressSpace(page);
  await settle();
  check(e.defaultPrevented && S.phase === "move" && S.throw === 3, "Space does not throw again before the move");
});

await scenario("undo", async (page) => {
  const { S, net } = page;
  await newGame(page, "human", "human");
  page.throws.push(3);
  pressSpace(page);
  await settle();
  check(S.phase === "move" && S.moves.length === 5, "undo: the opening throw 3 has five moves");
  const decision = { moves: JSON.stringify(S.moves), prob: S.prob };
  net.hold = (path) => path === "/api/analyze";
  click(square(page, S.moves[2].from));
  check(S.turn === "black" && S.history.length === 1, "undo: White's real choice is recorded");
  page.throws.push(2);
  pressSpace(page);
  await settle();
  check(S.phase === "throwing" && net.held.length === 2, "undo: the win chance and Black's throw are in flight");
  click(page.$("btn-undo"));
  await settle();
  check(net.aborted.length === 2, `undo aborts the requests in flight: ${net.aborted}`);
  net.hold = null;
  net.release();
  await settle();
  check(S.turn === "white" && S.phase === "move" && S.throw === 3, "undo returns to White's decision");
  check(JSON.stringify(S.moves) === decision.moves, "undo drops the stale throw reply");
  check(S.prob === decision.prob, "undo drops the stale win-chance reply");
  check(S.log.length === 0 && page.$("log").children.length === 0, "undo cuts back the record");

  // Replies that come despite the abort are dropped as well.
  net.hold = (path) => path === "/api/analyze";
  net.late = true;
  click(square(page, S.moves[2].from));
  page.throws.push(2);
  pressSpace(page);
  await settle();
  check(S.phase === "throwing" && net.held.length === 2, "undo: again, the win chance and Black's throw are in flight");
  click(page.$("btn-undo"));
  net.hold = null;
  net.release();
  await settle();
  check(S.turn === "white" && S.phase === "move" && JSON.stringify(S.moves) === decision.moves && S.prob === decision.prob,
    "undo drops the replies that come after all");
});

await scenario("new game", async (page) => {
  const { S, net } = page;
  await newGame(page, "random", "random");
  net.hold = aiRequest;
  check(await page.runUntil(() => net.held.length > 0), "new game: an AI throw is in flight");
  net.hold = null;
  click(page.$("btn-new"));
  await settle();
  check(net.aborted.length === 1, "new game aborts the throw in flight");
  net.release();
  await settle();
  check(S.phase === "throw" && S.throw === null && S.moves.length === 0 && S.log.length === 0,
    "new game drops the stale throw");
  check(await page.runUntil(() => S.winner !== null), "new game: the game after a dropped reply plays to the end");
});

await scenario("forced move", async (page) => {
  const { S } = page;
  await newGame(page, "human", "human");
  Object.assign(S, { white: [1], black: [20, 23] });
  page.throws.push(2); // 1→3 is White's only move
  pressSpace(page);
  await settle();
  check(S.phase === "move" && S.moves.length === 1, "forced move: White has one legal move");
  click(square(page, 1)); // played by hand before the auto-play delay is up
  page.throws.push(2); // two moves for Black: 20→22 and 23→25
  pressSpace(page);
  await settle();
  check(S.turn === "black" && S.phase === "move" && S.moves.length === 2, "forced move: Black has a real choice");
  await page.runUntil(() => false); // runs every timer left
  check(S.phase === "move" && S.log.length === 1, "a forced move played by hand cancels its auto-play");

  await newGame(page, "human", "human");
  Object.assign(S, { white: [1], black: [20, 23] });
  page.throws.push(2);
  pressSpace(page);
  await settle();
  check(message(page).startsWith("Only one legal move"), "forced move: its auto-play is announced");
  page.$("chk-auto").checked = false; // switched off before the auto-play delay is up
  page.$("chk-auto").fire("change");
  await page.runUntil(() => false);
  check(S.phase === "move" && S.log.length === 0 && message(page) === "",
    `a forced move is not auto-played once auto-play is off: "${message(page)}"`);
});

await scenario("win chance", async (page) => {
  const { S, net } = page;
  // The win chance of a new game is unavailable: none is shown, until a throw's analysis
  // reports it.
  net.fault = { when: probRequest, network: true };
  await newGame(page, "human", "human");
  check(S.prob === null && probLabel(page) === "White —" && page.$("probbar").classList.contains("unknown"),
    `an unavailable win chance is not shown: "${probLabel(page)}"`);
  check(message(page).startsWith("Win chance unavailable"), `an unavailable win chance is reported: "${message(page)}"`);
  page.throws.push(3);
  pressSpace(page);
  await settle();
  check(S.prob !== null && probLabel(page).endsWith("%") && message(page) === "",
    `a throw's analysis restores the win chance: "${probLabel(page)}"`);

  // An older request for the same board (a new game's start) is aborted.
  net.hold = probRequest;
  click(page.$("btn-new"));
  await settle();
  net.hold = null;
  click(page.$("btn-new"));
  await settle();
  check(net.aborted.length === 1 && S.prob !== null && message(page) === "",
    `a new win-chance request aborts the one before: "${message(page)}"`);

  // Its failure, if it comes all the same, is ignored.
  net.late = true;
  net.hold = probRequest;
  click(page.$("btn-new"));
  await settle();
  net.hold = null;
  click(page.$("btn-new"));
  await settle();
  const prob = S.prob;
  net.fault = { when: probRequest, network: true };
  net.release();
  await settle();
  check(S.prob === prob && prob !== null && message(page) === "",
    `the failure of an earlier game's win-chance request is ignored: "${message(page)}"`);

  // So is one that the throw's analysis answered first.
  net.hold = probRequest;
  click(page.$("btn-new"));
  await settle();
  net.hold = null;
  page.throws.push(3);
  pressSpace(page);
  await settle();
  net.fault = { when: probRequest, network: true };
  net.release();
  await settle();
  check(S.prob !== null && S.phase === "move" && message(page) === "",
    `a win-chance request that the throw answered first is ignored: "${message(page)}"`);
});

await scenario("accuracy", async (page) => {
  const { S, net } = page;
  await newGame(page, "human", "human");
  // White's two real choices are judged by different evaluators (the API here has only the
  // network: the first throw's analysis is relabelled), and White then wins.
  Object.assign(S, { white: [24, 26, 30], black: [1] });
  const judges = ["heuristic"];
  net.edit = (path, request, payload) => (throwRequest(path, request) ? { ...payload, source: judges.shift() ?? "net" } : payload);
  page.throws.push(1, 1, 5, 1, 5); // extra throws only: White moves until it has won
  pressSpace(page);
  await settle();
  check(S.moves.length === 3 && S.source === "heuristic", "accuracy: White's first choice is judged heuristically");
  click(square(page, 30)); // bears off
  pressSpace(page);
  await settle();
  check(S.moves.length === 2 && S.source === "net", "accuracy: White's second choice is judged by the network");
  click(square(page, 24));
  for (let i = 0; i < 3 && !S.winner; i++) { // forced moves, auto-played
    pressSpace(page);
    await settle();
    await page.runUntil(() => S.phase !== "move");
  }
  check(S.winner === "white", "accuracy: White wins");
  check(message(page).startsWith("Your accuracy — White vs the heuristic estimate: 1 real choice, ")
    && message(page).includes("; White vs the neural net: 1 real choice, "),
    `the accuracy report names the judge of each choice: "${message(page)}"`);
});

await scenario("players changed", async (page) => {
  const { S, net } = page;
  await newGame(page, "random", "human");
  page.throws.push(3); // five moves: no forced move to auto-play
  net.hold = (path) => path === "/api/ai";
  check(await page.runUntil(() => net.held.length > 0), "players changed: White's AI move is in flight");
  net.hold = null;
  choosePlayer(page, "white", "human");
  await settle();
  check(net.aborted.length === 1, "switching a thinking AI to a human aborts its move request");
  net.release();
  await settle();
  await page.runUntil(() => false); // runs every timer left
  check(S.turn === "white" && S.phase === "move" && S.throw === 3 && S.log.length === 0,
    "switching a thinking AI to a human drops its move");

  await newGame(page, "random", "human");
  net.hold = throwRequest;
  check(await page.runUntil(() => net.held.length > 0), "players changed: White's AI throw is in flight");
  net.hold = null;
  choosePlayer(page, "white", "net");
  net.release();
  await settle();
  check(S.phase === "throw" && S.throw === null, "switching bots takes back the throw in flight");
  check(await page.runUntil(() => S.log.length > 0), "the new bot throws and plays");

  await newGame(page, "human", "human");
  Object.assign(S, { white: [28], black: [1, 2] });
  page.throws.push(1); // a piece on 28 bears off only with a 3
  pressSpace(page);
  await settle();
  check(S.phase === "pass" && S.log.length === 1, "players changed: White's turn passes");
  choosePlayer(page, "black", "random");
  check(S.turn === "black" && S.phase !== "pass", "a player change during a pass ends the pass");
  check(await page.runUntil(() => S.log.length > 1), "Black's new AI throws and plays");
});

await scenario("players changed between AIs", async (page) => {
  const { S, net } = page;
  // Between two AIs the moves come with the mover's choice, and no win chance of theirs.
  page.throws.push(3); // five moves: no forced move to auto-play
  await newGame(page, "random", "random");
  check(await page.runUntil(() => S.phase === "move"), "between AIs: a throw's moves are in");
  const { turn, throw: t, log } = { ...S, log: S.log.length };
  check(S.choice !== null && S.moves.every((m) => !("best" in m)), "between AIs: the moves come from the AI's reply");
  choosePlayer(page, turn, "human");
  await settle();
  check(S.phase === "move" && S.turn === turn && S.throw === t && S.choice === null && S.moves.every((m) => "best" in m),
    "a human taking over from an AI gets the same throw, analyzed");
  await page.runUntil(() => false); // runs every timer left
  check(S.log.length === log && S.phase === "move", "the old AI's choice is not played for the human");

  await newGame(page, "random", "random");
  check(await page.runUntil(() => S.phase === "move"), "between AIs: again, a throw's moves are in");
  net.requests.length = 0;
  choosePlayer(page, S.turn, "net");
  await settle();
  check(net.requests.length === 1 && net.requests[0][0] === "/api/ai" && net.requests[0][1].bot === "net",
    `a new AI is asked about the same throw: ${JSON.stringify(net.requests)}`);
  check(await page.runUntil(() => S.winner !== null), "the game between the new AIs plays to the end");
});

await scenario("failures", async (page) => {
  const { S, net } = page;
  await newGame(page, "human", "human");
  // The human's throw fails (HTTP error): it is taken back and can be thrown again.
  net.fault = { when: throwRequest };
  page.throws.push(3);
  pressSpace(page);
  await settle();
  check(S.phase === "throw" && S.throw === null && S.faces === null, "a human's failed throw is taken back");
  check(message(page) === "Throw failed (injected failure) — throw again.", `a human's failed throw is reported: "${message(page)}"`);
  check(!page.$("btn-throw").disabled && page.$("throw-value").innerHTML === "&nbsp;", "the human can throw again");
  // A reply that is not JSON, such as the server's to a request that timed out.
  net.fault = { when: throwRequest, plain: [408, "Request Timeout", "request timed out"] };
  page.throws.push(3);
  click(page.$("btn-throw"));
  await settle();
  check(S.phase === "throw" && message(page) === "Throw failed (408 Request Timeout) — throw again.",
    `a plain-text error reply is reported: "${message(page)}"`);
  page.throws.push(3);
  click(page.$("btn-throw"));
  await settle();
  check(S.phase === "move" && message(page) === "", "the human's next throw works");
  // White becomes an AI whose move request fails (network error): it is retried.
  net.fault = { when: (path) => path === "/api/ai", network: true };
  choosePlayer(page, "white", "random");
  await settle();
  check(S.phase === "move" && message(page).startsWith("AI move failed (Failed to fetch) — retrying in "),
    `a failed AI move is reported: "${message(page)}"`);
  check(await page.runUntil(() => S.log.length === 1), "the failed AI move is retried and played");
  check(!message(page).includes("failed"), "the failure report is cleared after the retry");
  // Black becomes an AI whose throw (against White's AI: only Black's AI is asked) fails: it
  // is taken back and retried.
  net.fault = { when: aiRequest, network: true };
  choosePlayer(page, "black", "random");
  check(await page.runUntil(() => message(page).includes("failed")), "Black's AI throw fails");
  check(S.turn === "black" && S.phase === "throw" && S.throw === null, "a failed AI throw is taken back");
  check(await page.runUntil(() => S.log.length === 2), "the AI throws again and plays");
  check(!message(page).includes("failed"), "the failure report is cleared after the AI's new throw");
  // An AI replaced by a human during its retry wait takes its retry notice with it.
  await newGame(page, "random", "human");
  net.fault = { when: throwRequest, network: true };
  check(await page.runUntil(() => message(page).includes("retrying")), "White's AI throw fails");
  choosePlayer(page, "white", "human");
  check(message(page) === "", `switching the AI to a human clears its retry notice: "${message(page)}"`);
});

await scenario("timeouts", async (page) => {
  const { S, net } = page;
  await newGame(page, "human", "human");
  // The human's throw is never analyzed: it times out, and can be thrown again.
  net.hold = throwRequest;
  page.throws.push(3);
  pressSpace(page);
  await settle();
  check(S.phase === "throwing" && net.held.length === 1, "timeouts: the human's throw is in flight");
  check(await page.runUntil(() => S.phase !== "throwing"), "timeouts: a throw without a reply times out");
  check(net.aborted.length === 1, "a timed-out request is aborted");
  check(S.phase === "throw" && message(page) === "Throw failed (no reply within 30 s) — throw again.",
    `a timed-out throw is reported: "${message(page)}"`);
  net.hold = null;
  page.throws.push(3);
  click(page.$("btn-throw"));
  await settle();
  check(S.phase === "move" && message(page) === "", "timeouts: the human's next throw works");
  // White becomes an AI whose move never comes: it times out and is retried.
  net.hold = (path) => path === "/api/ai";
  choosePlayer(page, "white", "random");
  check(await page.runUntil(() => message(page).startsWith("AI move failed (no reply within 30 s) — retrying in ")),
    `a timed-out AI move is reported: "${message(page)}"`);
  net.hold = null;
  check(await page.runUntil(() => S.log.length === 1), "the timed-out AI move is retried and played");
  check(net.aborted.length === 2 && !message(page).includes("failed"), "timeouts: the AI recovers");
});

await scenario("offline", async (page) => {
  check(page.$("engine-status").textContent.startsWith("Engine offline"), "offline: the header says so");
  const controls = ["btn-throw", "btn-new", "btn-undo", "sel-white", "sel-black", "chk-hints", "chk-auto", "rng-speed"];
  check(controls.every((id) => page.$(id).disabled), "offline: the panel's controls are disabled");
  check(!page.document.listeners.keydown, "offline: Space is not bound");
}, { offline: true });

for (const f of failures.slice(0, 10)) console.log("FAIL", f);
console.log(`web app: ${checks} checks, ${failures.length} failed (${games} AI-vs-AI games, ${entries} record entries)`);
process.exit(failures.length ? 1 : 0);
