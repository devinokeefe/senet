// The portable single-file page (built by tools/build_portable.py) as a browser runs it: its
// inline scripts, in order, on its own markup, in node:vm with the DOM stub and fake timers
// of tools/page_stub.mjs, and no network. The page must load nothing from elsewhere, embed
// the given network exactly, and play AI-vs-AI games with it to the end with no request to
// a server and no error. The games start late: the page's search with the real network
// takes about two seconds a throw here.
// Usage: node tools/check_portable.mjs dist/senet-portable.html models/senet_net.bin
import { readFileSync } from "node:fs";
import { Document, loadPage, newGame, seeded } from "./page_stub.mjs";
import { readSnn1 } from "./snn1.mjs";

// The players, and the position each game starts from (White to throw).
const GAMES = [
  ["net:1", "net", { white: [20, 23, 25], black: [21, 24, 26] }],
  ["random", "net:1", { white: [26, 28], black: [22, 29] }],
];

const [pagePath, netPath] = process.argv.slice(2);
if (!pagePath || !netPath) {
  console.error("usage: node tools/check_portable.mjs <page.html> <net.bin>");
  process.exit(2);
}

let checks = 0, games = 0, entries = 0;
const failures = [];
function check(ok, what) {
  checks++;
  if (!ok) failures.push(what);
}
process.on("unhandledRejection", (e) => check(false, `unhandled rejection: ${e?.stack ?? e}`));

try {
  const html = readFileSync(pagePath, "utf8");
  const markup = new Document(html);
  const scripts = markup.querySelectorAll("script");
  check(scripts.length > 0 && scripts.every((s) => !("src" in s.attributes)), "the page's scripts are all inline");
  check(markup.querySelectorAll("link").length === 0, "the page links nothing: its style is inline");

  const random = seeded(7);
  const requests = [], logged = [];
  const log = (...args) => logged.push(args.join(" "));
  const page = await loadPage(html, scripts.map((s, i) => [`${pagePath} script ${i + 1}`, s.text]), {
    random,
    fetch: async (path) => {
      requests.push(path);
      throw new TypeError("Failed to fetch");
    },
    crypto: {
      getRandomValues(buf) {
        for (let i = 0; i < buf.length; i++) buf[i] = Math.floor(random() * 256);
        return buf;
      },
    },
    console: { log, info: log, warn: log, error: log, debug: log },
  });
  check(page.evaluate("JSON.stringify(window.SENET_NET)") === JSON.stringify(readSnn1(netPath)),
    `the page embeds ${netPath} exactly`);
  check(await page.runUntil(() => page.evaluate("INFO") !== null), "the page starts");
  check(page.$("engine-status").textContent.startsWith("Portable AI"), "the page's engine is its own");

  const S = page.evaluate("S"); // the game state, a top-level const of app.js
  const other = (c) => (c === "white" ? "black" : "white");
  for (const [white, black, start] of GAMES) {
    const what = `${white} vs ${black}`;
    await newGame(page, white, black);
    Object.assign(S, start); // before White's first throw
    const ended = await page.runUntil(() => S.winner !== null);
    check(ended, `${what}: the game ends`);
    if (!ended) continue;
    games++;
    entries += S.log.length;
    check(S[S.winner].length === 0 && S[other(S.winner)].length > 0, `${what}: the winner has borne off every piece`);
    check(S.prob !== null, `${what}: the win chance is shown`);
    const message = page.$("message").textContent;
    check(!/failed|unavailable/i.test(message), `${what}: no request failed: "${message}"`);
  }
  check(requests.length === 0, `the page asks no server: it requested ${requests.join(", ")}`);
  check(logged.length === 0, `the page logs nothing: ${logged.join("; ")}`);
} catch (e) {
  check(false, e?.stack ?? String(e));
}

for (const f of failures.slice(0, 10)) console.log("FAIL", f);
console.log(`portable page: ${checks} checks, ${failures.length} failed (${games} AI-vs-AI games, ${entries} record entries)`);
process.exit(failures.length ? 1 : 0);
