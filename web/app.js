"use strict";
// Senet UI. The server is the rules authority: every throw is sent to /api/analyze,
// which returns the legal moves (with resulting boards and win chances); the client
// only keeps the game state, draws it and asks /api/ai for computer moves.
// tools/check_web_app.mjs plays it in node against the portable build's API.

const BOT_LABELS = {
  "perfect": "Perfect (solved)",
  "net:1": "Neural net + search",
  "net": "Neural net",
  "expectimax:3": "Expectimax (3 throws)",
  "expectimax:2": "Expectimax (2 throws)",
  "greedy": "Greedy heuristic",
  "random": "Random",
};
const SPECIAL = {
  15: { glyph: "☥", name: "House of Rebirth" },
  26: { glyph: "𓄤", name: "House of Beauty — must land here exactly" },
  27: { glyph: "𓈗", name: "House of Water — back to 15, or the highest empty square below" },
  28: { glyph: "III", name: "House of Three Truths — bear off with 3" },
  29: { glyph: "II", name: "House of Re-Atoum — bear off with 2" },
  30: { glyph: "𓂀", name: "House of Horus — bear off with 1" },
};
const PIECE_PATH = {
  white: "M50 8 C58 8 62 14 64 22 L78 86 Q79 92 72 92 L28 92 Q21 92 22 86 L36 22 C38 14 42 8 50 8 Z",
  black: "M24 10 H76 Q82 10 80 18 L64 50 L80 82 Q82 90 76 90 H24 Q18 90 20 82 L36 50 L20 18 Q18 10 24 10 Z",
};
const SQUARES = 30;
const OFF = 31; // "square" of a borne-off piece
const WATER = 27;
const FORCED_MOVE_DELAY_MS = 450; // before a human's only legal move is played for them
const MIN_PASS_PAUSE_MS = 700; // a turn without a legal move stays on screen at least this long
const AI_RETRY_MS = 2000; // before a failed request on the AI's turn is repeated
const REQUEST_TIMEOUT_MS = 30000; // a request without a reply by then fails
const MIN_SHOWN_LOSS = 0.0005; // smallest win chance given away that the record shows: it shows as 0.1%
// What judged the win chances, by the `source` of an analysis.
const JUDGES = { perfect: "perfect play", net: "the neural net", heuristic: "the heuristic estimate" };

const $ = (id) => document.getElementById(id);
const other = (c) => (c === "white" ? "black" : "white");
const cap = (c) => c[0].toUpperCase() + c.slice(1);
const pct = (p) => (p * 100).toFixed(1) + "%";
const botLabel = (bot) => BOT_LABELS[bot] || bot;

let INFO = null;

// Everything a game consists of; New game starts from a fresh copy.
function freshGame(start = { white: [], black: [], turn: "white" }) {
  return {
    white: start.white.slice(), black: start.black.slice(), turn: start.turn,
    // "throw": the mover must throw; "throwing": the throw is being analyzed; "move":
    // `throw`/`moves` hold the throw to play; "pass": the throw has no legal move and
    // the turn passes shortly; "over".
    phase: "throw", throw: null, faces: null, moves: [],
    // Between two AIs, `moves` come from the mover's /api/ai reply, with its `choice`;
    // otherwise from /api/analyze, and `choice` is null.
    choice: null,
    // White's win chance before the throw (null: unknown) and its `source`, which also
    // judged the win chances of `moves` from /api/analyze.
    winner: null, last: null, prob: 0.5, source: "",
    // A human's real choices and the win chance they gave away, per color and per judge:
    // { white: { perfect: { decisions, loss }, ... }, black: { ... } }.
    history: [], log: [], acc: { white: {}, black: {} },
  };
}
// `gen` is bumped whenever the flow is redirected (a move is played, new game, undo,
// players changed); timers and replies from an older generation are then ignored.
const S = { ...freshGame(), hover: null, gen: 0 };
// Aborts the requests for the turn in progress (a throw's analysis, an AI move) on the next
// redirection.
let turnRequests = new AbortController();
function redirect() {
  S.gen++;
  turnRequests.abort();
  turnRequests = new AbortController();
}

// Undo returns to a human decision: the position and the throw being decided.
const DECISION_KEYS = ["white", "black", "turn", "throw", "faces", "moves", "last", "prob", "source"];
function snapshot() {
  const h = { acc: structuredClone(S.acc), logLen: S.log.length };
  for (const k of DECISION_KEYS) h[k] = S[k];
  return h;
}
function restore(h) {
  for (const k of DECISION_KEYS) S[k] = h[k];
  Object.assign(S, { acc: h.acc, phase: "move", choice: null, winner: null });
  S.log.length = h.logLen;
}
function clearThrow() {
  Object.assign(S, { throw: null, faces: null, moves: [], choice: null });
}
// Takes back a throw whose analysis will not be used.
function abandonThrow() {
  if (S.phase !== "throwing") return;
  S.phase = "throw";
  clearThrow();
}

// Resolves once the browser has painted the page as it is now.
function afterPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
}

// Sends a request, which is aborted with `signal` (its reply is no longer wanted), and fails
// without a reply within REQUEST_TIMEOUT_MS.
async function api(path, body, signal) {
  // The portable single-file build answers the same API in the browser, on this thread:
  // the page is painted first (the sticks start rolling), and an aborted request not run.
  if (window.SENET_LOCAL_API) {
    await afterPaint();
    signal?.throwIfAborted();
    return window.SENET_LOCAL_API(path, body);
  }
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort);
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS);
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  try {
    const r = await fetch(path, { ...opts, signal: controller.signal });
    const text = await r.text();
    let j = null;
    try {
      j = JSON.parse(text);
    } catch {
      // not JSON, such as the server's plain-text reply to a request that timed out
    }
    if (!r.ok) throw new Error(j?.error ?? `${r.status} ${r.statusText}`.trim());
    if (j === null) throw new Error("the reply is not JSON");
    return j;
  } catch (e) {
    throw timedOut ? new Error(`no reply within ${REQUEST_TIMEOUT_MS / 1000} s`) : e;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}

// The "AI speed" slider runs slow (left) to fast (right); the pause is its mirror image.
function aiDelay() {
  const r = $("rng-speed");
  return Number(r.max) + Number(r.min) - Number(r.value);
}
function botOf(color) { return $("sel-" + color).value; }
function isAI(color) { return botOf(color) !== "human"; }
function canHumanThrow() { return S.phase === "throw" && !isAI(S.turn); }
function humanMoving() { return S.phase === "move" && !isAI(S.turn); }
function later(fn, ms) {
  const gen = S.gen;
  setTimeout(() => { if (gen === S.gen) fn(); }, ms);
}
// Assign only on change, so that live regions announce real updates only.
function setText(el, text) { if (el.textContent !== text) el.textContent = text; }
function setHTML(el, html) { if (el.innerHTML !== html) el.innerHTML = html; }
function setMessage(t) { setText($("message"), t || ""); }

// ---------- Rendering ----------

function squareCell(s) {
  if (s <= 10) return { row: 0, col: s - 1, arrow: "→" };
  if (s <= 20) return { row: 1, col: 20 - s, arrow: "←" };
  return { row: 2, col: s - 21, arrow: "→" };
}

function pieceSvg(color) {
  // Written as the browser serializes it, so that setHTML sees an unchanged piece.
  return `<svg class="piece ${color}" viewBox="0 0 100 100" aria-hidden="true"><path d="${PIECE_PATH[color]}"></path></svg>`;
}
function pieceOn(s) {
  return S.white.includes(s) ? "white" : S.black.includes(s) ? "black" : null;
}

function span(className, text = "") {
  const el = document.createElement("span");
  el.className = className;
  el.textContent = text;
  return el;
}

function buildBoard() {
  // Squares are buttons in track order (1..30), placed on the S-shaped grid by CSS. Their
  // number and marks never change; render() fills the piece spot and the accessible state.
  const board = $("board");
  const hover = (s) => () => { S.hover = s; renderHighlights(); };
  for (let s = 1; s <= SQUARES; s++) {
    const { row, col, arrow } = squareCell(s);
    const el = document.createElement("button");
    el.type = "button";
    el.className = "sq" + ((row + col) % 2 ? " alt" : "") + (SPECIAL[s] ? " special" : "");
    el.style.gridRow = row + 1;
    el.style.gridColumn = col + 1;
    el.dataset.sq = s;
    el.title = SPECIAL[s] ? `${s}: ${SPECIAL[s].name}` : `Square ${s}`;
    el.tabIndex = -1;
    el.setAttribute("aria-disabled", "true");
    el.append(span("spot"), span("num", s));
    if (s % 10 === 1) el.append(span("arrow", arrow)); // the first square of each row
    if (SPECIAL[s]) el.append(span("glyph", SPECIAL[s].glyph));
    el.addEventListener("click", () => onSquareClick(s));
    el.addEventListener("mouseenter", hover(s));
    el.addEventListener("mouseleave", hover(null));
    el.addEventListener("focus", hover(s));
    el.addEventListener("blur", hover(null));
    board.appendChild(el);
  }
  // A bear-off row is the target of the mover's bear-off move, like a board square.
  for (const c of ["white", "black"]) {
    const row = $("off-" + c);
    const ifMover = (fn) => () => { if (c === S.turn) fn(); };
    row.addEventListener("click", ifMover(() => onSquareClick(OFF)));
    row.addEventListener("mouseenter", ifMover(hover(OFF)));
    row.addEventListener("mouseleave", hover(null));
  }
}

function render() {
  renderBoard();
  renderTurnCard();
  renderProb();
  renderHighlights();
  renderMoveList();
  renderLog();
}

function renderBoard() {
  const moving = humanMoving();
  const hints = $("chk-hints").checked;
  for (const el of $("board").children) {
    const s = Number(el.dataset.sq);
    const piece = pieceOn(s);
    setHTML(el.querySelector(".spot"), piece ? pieceSvg(piece) : "");
    // Keyboard: Tab reaches the pieces that can move; any move square can be activated.
    const move = moving ? moveAt(s) : null;
    el.setAttribute("aria-label", squareLabel(s, piece, move, hints));
    el.setAttribute("aria-disabled", String(!move));
    el.tabIndex = move && move.from === s ? 0 : -1;
  }
  for (const c of ["white", "black"]) {
    const off = INFO.start[c].length - S[c].length;
    setHTML($("off-" + c), `<span>${cap(c)} borne off:</span><span class="sr-only">${off}</span>` + pieceSvg(c).repeat(off));
  }
}

function renderTurnCard() {
  $("turn-dot").className = "dot" + (S.turn === "black" ? " black" : "");
  const who = isAI(S.turn) ? `${cap(S.turn)} (${botLabel(botOf(S.turn))})` : `${cap(S.turn)} (you)`;
  setText($("turn-text"), S.winner
    ? `${cap(S.winner)} wins!`
    : S.phase === "move" ? `${who} — choose a move` : `${who} to throw`);
  const sticks = $("sticks").children;
  for (let i = 0; i < 4; i++) {
    sticks[i].className = "stick" + (S.faces ? (S.faces[i] ? " light" : " dark") : "");
  }
  const again = S.phase === "move" && INFO.extra_throws.includes(S.throw) ? " <small>— throw again after moving</small>" : "";
  setHTML($("throw-value"), S.throw ? `${S.throw}${again}` : "&nbsp;");
  $("btn-throw").disabled = !canHumanThrow();
  $("btn-undo").disabled = S.history.length === 0;
}

function renderProb() {
  const p = S.winner ? (S.winner === "white" ? 1 : 0) : S.prob;
  const known = p !== null;
  const bar = $("probbar");
  bar.classList.toggle("unknown", !known);
  $("prob-white").style.width = known ? (p * 100).toFixed(2) + "%" : "50%";
  $("prob-white-label").textContent = `White ${known ? pct(p) : "—"}`;
  $("prob-black-label").textContent = `${known ? pct(1 - p) : "—"} Black`;
  bar.setAttribute("aria-valuenow", known ? (p * 100).toFixed(1) : "50");
  bar.setAttribute("aria-valuetext", known ? `White ${pct(p)}, Black ${pct(1 - p)}` : "unknown");
  const caption = {
    perfect: "Chance of winning with perfect play from here, before the throw (solved database).",
    net: "Neural-net estimate of the chance of winning from here, before the throw.",
    heuristic: "Rough heuristic estimate of the chance of winning (no database or network for this position).",
  }[S.source] || "";
  $("prob-source").textContent = S.winner ? "" : caption;
}

// Square a move visibly lands on: OFF when borne off, 27 for a drowning (the piece then returns to 15 or below).
function targetOf(m) { return m.kind === "off" ? OFF : m.kind === "water" ? WATER : m.to; }
// The move starting or landing on square s (at most one: each piece has one move per throw).
function moveAt(s) { return S.moves.find((m) => m.from === s || targetOf(m) === s); }

// "9→12 (swap)"; with arrow = " to " it reads well as an accessible name.
function describeMove(m, arrow = "→") {
  if (m.kind === "water") return `${m.from}${arrow}${WATER} (drowned ${arrow.trim()} ${m.to})`;
  const dest = m.kind === "off" ? "off" : m.to;
  return `${m.from}${arrow}${dest}${m.dir === "back" ? " backward" : ""}${m.kind === "swap" ? " (swap)" : ""}`;
}

function squareLabel(s, piece, move, hints) {
  let label = `Square ${s}${SPECIAL[s] ? ` (${SPECIAL[s].name})` : ""}, ${piece ? piece + " piece" : "empty"}`;
  if (move) {
    label += `; ${move.from === s ? "play" : "target of"} ${describeMove(move, " to ")}`;
    if (hints) label += `, wins ${pct(move.win_prob)}`;
  }
  return label;
}

function renderHighlights() {
  const moving = humanMoving();
  const hints = $("chk-hints").checked;
  const hovered = moving ? moveAt(S.hover) : null;
  for (const el of $("board").children) {
    const s = Number(el.dataset.sq);
    el.classList.remove("movable", "target", "selected", "last");
    el.querySelector(".badge")?.remove();
    if (S.last && (S.last.from === s || S.last.to === s)) el.classList.add("last");
    if (!moving) continue;
    const m = S.moves.find((mv) => mv.from === s);
    if (m) {
      el.classList.add("movable");
      if (hints) el.append(span("badge" + (m.best ? " best" : ""), pct(m.win_prob)));
    }
    if (hovered && hovered.from === s) el.classList.add("selected");
    if (hovered && targetOf(hovered) === s) el.classList.add("target");
  }
  const bearOff = moving && S.moves.some((m) => m.kind === "off");
  for (const c of ["white", "black"]) {
    const row = $("off-" + c);
    row.classList.toggle("reachable", bearOff && c === S.turn);
    row.classList.toggle("target", hovered?.kind === "off" && c === S.turn);
  }
}

function renderMoveList() {
  const show = $("chk-hints").checked && humanMoving();
  $("hint-card").hidden = !show;
  if (!show) return;
  const ol = $("move-list");
  ol.innerHTML = "";
  S.moves
    .map((m, i) => ({ m, i }))
    .sort((a, b) => b.m.win_prob - a.m.win_prob)
    .forEach(({ m, i }) => {
      const li = document.createElement("li");
      if (m.best) li.className = "best";
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = `${describeMove(m)} — wins ${pct(m.win_prob)}`;
      b.addEventListener("click", () => applyMove(i));
      li.appendChild(b);
      ol.appendChild(li);
    });
}

// Between two renders the record only grows, or is cut back by undo or a new game.
function renderLog() {
  const ol = $("log");
  while (ol.children.length > S.log.length) ol.lastElementChild.remove();
  if (ol.children.length === S.log.length) return;
  for (const e of S.log.slice(ol.children.length)) {
    const li = document.createElement("li");
    li.className = e.color === "white" ? "w" : "b";
    li.innerHTML = e.html;
    ol.appendChild(li);
  }
  ol.scrollTop = ol.scrollHeight; // follow new entries
}

// ---------- Game flow ----------
// throw (doThrow) → move (human click / requestAIMove → applyMove) → throw again or the
// other side's throw; a throw without legal moves goes through "pass" (endPass).

// Requests for the win chance are numbered, and only the reply to the latest one is used:
// replies can come out of order. A request is retired, and aborted, by the next one, an
// undo (which restores the win chance with the position), the end of the game and a
// throw's analysis (which reports the win chance too).
let probRequest = 0;
let probRequests = new AbortController();
function retireProbRequest() {
  probRequest++;
  probRequests.abort();
  probRequests = new AbortController();
}

// Asks for the win chance before the throw.
async function refreshProb() {
  if (S.winner) return;
  retireProbRequest();
  const request = probRequest;
  try {
    const r = await api("/api/analyze", { white: S.white, black: S.black, turn: S.turn }, probRequests.signal);
    if (request !== probRequest) return;
    Object.assign(S, { prob: r.white_win_prob, source: r.source });
  } catch (e) {
    if (request !== probRequest) return;
    Object.assign(S, { prob: null, source: "" });
    setMessage(`Win chance unavailable (${e.message}).`);
  }
  renderProb();
}

function throwSticks() {
  const buf = new Uint8Array(1);
  crypto.getRandomValues(buf);
  const faces = [0, 1, 2, 3].map((i) => ((buf[0] >> i) & 1) === 1);
  const light = faces.filter(Boolean).length;
  return { faces, value: light === 0 ? 5 : light };
}

// Renders, then schedules whatever happens next without a human click.
function step() {
  render();
  if (S.winner) return;
  if (isAI(S.turn)) {
    if (S.phase === "throw") later(doThrow, aiDelay());
    else if (S.phase === "move" && S.choice !== null) later(() => applyMove(S.choice), aiDelay());
    else if (S.phase === "move") requestAIMove();
  } else if (S.phase === "move" && S.moves.length === 1 && $("chk-auto").checked) {
    setMessage("Only one legal move — played automatically.");
    later(playForcedMove, FORCED_MOVE_DELAY_MS);
  }
}

// Plays a human's only legal move, unless auto-play was switched off meanwhile.
function playForcedMove() {
  if ($("chk-auto").checked) applyMove(0);
  else setMessage("");
}

// A request for the current turn failed: take back the throw being analyzed and say so.
// A human just throws again; on the AI's turn the request is repeated after AI_RETRY_MS.
function requestFailed(what, e) {
  abandonThrow();
  const ai = isAI(S.turn);
  setMessage(`${what} failed (${e.message}) — ${ai ? `retrying in ${AI_RETRY_MS / 1000} s` : "throw again"}.`);
  render();
  if (ai) later(step, AI_RETRY_MS);
}

function doThrow() {
  if (S.phase !== "throw") return;
  const { faces, value } = throwSticks();
  Object.assign(S, { phase: "throwing", faces, throw: value });
  setMessage("");
  const st = $("sticks");
  st.classList.remove("rolling");
  void st.offsetWidth; // restart the animation
  st.classList.add("rolling");
  render();
  fetchMoves();
}

// Asks for the moves of the throw being analyzed. Between two AIs, whose moves' win chances
// are never shown, the mover's AI is asked directly.
async function fetchMoves() {
  const gen = S.gen;
  const twoAIs = isAI("white") && isAI("black");
  const position = { white: S.white, black: S.black, turn: S.turn, throw: S.throw };
  let r;
  try {
    r = await (twoAIs ? askAI() : api("/api/analyze", position, turnRequests.signal));
  } catch (e) {
    if (gen === S.gen) requestFailed("Throw", e);
    return;
  }
  if (gen !== S.gen) return;
  if (twoAIs) {
    S.choice = r.choice;
  } else {
    retireProbRequest();
    Object.assign(S, { prob: r.white_win_prob, source: r.source });
  }
  S.moves = r.moves;
  if (S.moves.length) {
    S.phase = "move";
    step();
    return;
  }
  S.phase = "pass";
  S.log.push({ color: S.turn, html: `${cap(S.turn)} threw ${S.throw}: no legal move, turn passes` });
  setMessage(`${cap(S.turn)} has no legal move with a ${S.throw} — the turn passes.`);
  render();
  later(endPass, Math.max(MIN_PASS_PAUSE_MS, aiDelay()));
}

function endPass() {
  S.turn = other(S.turn);
  S.phase = "throw";
  clearThrow();
  refreshProb();
  step();
}

// The reply of the mover's AI for the current throw.
function askAI() {
  return api("/api/ai", {
    white: S.white, black: S.black, turn: S.turn, throw: S.throw, bot: botOf(S.turn),
    seed: Math.floor(Math.random() * 1e9),
  }, turnRequests.signal);
}

async function requestAIMove() {
  const gen = S.gen;
  let a;
  try {
    a = await askAI();
  } catch (e) {
    if (gen === S.gen) requestFailed("AI move", e);
    return;
  }
  if (gen !== S.gen) return;
  setMessage(""); // clears the report of a failed attempt
  // Play the chosen move from S.moves, whose win chances (from /api/analyze) are the
  // reference for the game record. A piece has at most one move per throw.
  const from = a.moves[a.choice].from;
  later(() => applyMove(S.moves.findIndex((m) => m.from === from)), aiDelay());
}

function applyMove(i) {
  const m = S.moves[i];
  if (S.phase !== "move" || !m) return;
  redirect(); // this throw is used up: drop timers still aimed at it (e.g. a forced-move auto-play)
  recordMove(m);
  S.white = m.white;
  S.black = m.black;
  S.last = { from: m.from, to: m.kind === "off" ? null : m.to };
  clearThrow();
  if (m.winner) {
    finishGame(m.winner);
    return;
  }
  S.turn = m.next_turn;
  S.phase = "throw";
  refreshProb();
  step();
}

// Adds the move to the game record and, for a human's real choice, to Undo and the
// accuracy tally.
function recordMove(m) {
  const human = !isAI(S.turn);
  const choices = S.moves.length;
  const loss = Math.max(...S.moves.map((x) => x.win_prob)) - m.win_prob;
  if (human && choices > 1) {
    S.history.push(snapshot());
    const tally = (S.acc[S.turn][S.source] ??= { decisions: 0, loss: 0 });
    tally.decisions += 1;
    tally.loss += loss;
  }
  // Win chance given away is shown for humans, and for an AI playing a human.
  const shown = choices > 1 && loss >= MIN_SHOWN_LOSS && (human || !isAI(other(S.turn)));
  const note = shown ? ` <span class="loss">(−${(loss * 100).toFixed(1)}% vs best)</span>` : "";
  S.log.push({ color: S.turn, html: `${cap(S.turn)} ${S.throw}: ${describeMove(m)}${note}` });
}

function finishGame(winner) {
  retireProbRequest();
  S.winner = winner;
  S.phase = "over";
  S.log.push({ color: winner, html: `<b>${cap(winner)} bears off the last piece and wins.</b>` });
  const parts = ["white", "black"].flatMap((c) => Object.entries(S.acc[c]).map(([source, t]) =>
    `${cap(c)} vs ${JUDGES[source] ?? source}: ${t.decisions} real choice${t.decisions === 1 ? "" : "s"}, ` +
    `${(t.loss * 100).toFixed(1)}% total win chance given away`));
  setMessage(parts.length ? `Your accuracy — ${parts.join("; ")}.` : "");
  render();
}

function onSquareClick(s) {
  const m = humanMoving() ? moveAt(s) : null;
  if (m) applyMove(S.moves.indexOf(m));
}

function newGame() {
  redirect();
  Object.assign(S, freshGame(INFO.start));
  setMessage("");
  refreshProb();
  step();
}

function undo() {
  const h = S.history.pop();
  if (!h) return;
  redirect();
  retireProbRequest();
  restore(h);
  setMessage("Back to your last decision (same throw).");
  step();
}

// A player was switched (human ↔ AI or another bot): drop the old set-up's pending
// throw or move and continue the same game with the new one.
function onPlayersChanged() {
  redirect();
  setMessage(""); // e.g. a pending "retrying in 2 s" that no longer applies
  if (S.phase === "pass") {
    endPass(); // its timer was just cancelled
    return;
  }
  if (S.phase === "move" && S.choice !== null) {
    // The moves came with the old AI's choice: ask again about the same throw.
    Object.assign(S, { phase: "throwing", moves: [], choice: null });
    render();
    fetchMoves();
    return;
  }
  abandonThrow();
  step();
}

// ---------- Set-up ----------

// Space throws the sticks, except on controls where Space has its own meaning: form
// fields, and buttons/links the user reached with the keyboard (a button that merely
// kept focus after a mouse click does not swallow the shortcut).
let pointerFocus = false;
function spaceBelongsTo(el) {
  if (!(el instanceof Element)) return false;
  if (el.closest("input, select, textarea")) return true;
  const control = el.closest("button, summary, a[href]");
  return !!control && !pointerFocus && control.getAttribute("aria-disabled") !== "true";
}
function onKeyDown(e) {
  if (e.key === "Tab") pointerFocus = false;
  if (e.code !== "Space" || spaceBelongsTo(e.target)) return;
  e.preventDefault();
  if (!e.repeat && canHumanThrow()) doThrow();
}

// The header's line about the server's engine. The portable build has no server and says nothing.
function engineStatus() {
  if (window.SENET_LOCAL_API) return "";
  const db = INFO.database;
  if (db) {
    const kind = db.complete ? "perfect play (fully solved)" : "partially solved database";
    return `Engine: <b>${kind}</b>` + (INFO.network ? " · neural net loaded" : "");
  }
  return INFO.network ? "Engine: <b>neural net</b> (no database)" : "Engine: heuristic only";
}

// Whether the engine can play this bot: net bots need the network, "perfect" the full database.
function canPlay(bot) {
  if (bot.startsWith("net")) return INFO.network;
  if (bot === "perfect") return Boolean(INFO.database?.complete);
  return true;
}

function setUpPlayers() {
  const bots = INFO.bots.filter(canPlay);
  for (const c of ["white", "black"]) {
    const sel = $("sel-" + c);
    for (const b of bots) {
      const o = document.createElement("option");
      o.value = b;
      o.textContent = botLabel(b);
      sel.appendChild(o);
    }
    sel.addEventListener("change", onPlayersChanged);
  }
  if (bots.length) $("sel-black").value = bots[0]; // the API lists bots strongest first
}

function bindControls() {
  const speed = $("rng-speed");
  const describeSpeed = () => speed.setAttribute("aria-valuetext", `${(aiDelay() / 1000).toFixed(1)} s per AI action`);
  speed.addEventListener("input", describeSpeed);
  describeSpeed();
  $("btn-throw").addEventListener("click", doThrow);
  $("btn-new").addEventListener("click", newGame);
  $("btn-undo").addEventListener("click", undo);
  $("chk-hints").addEventListener("change", render);
  document.addEventListener("pointerdown", () => { pointerFocus = true; }, true);
  document.addEventListener("keydown", onKeyDown);
}

function showOffline() {
  $("engine-status").textContent = "Engine offline — start it with: senet serve";
  setText($("turn-text"), "Engine offline");
  for (const el of $("panel").querySelectorAll("button, input, select")) el.disabled = true;
}

async function init() {
  buildBoard();
  try {
    INFO = await api("/api/info");
  } catch {
    showOffline();
    return;
  }
  $("engine-status").innerHTML = engineStatus();
  setUpPlayers();
  bindControls();
  newGame();
}

init();
