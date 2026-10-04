# Interfaces

The engine has four interfaces. Each is defined in one place; where it is also
implemented elsewhere, a check keeps the implementations in agreement:

| Interface | Defined in | Also implemented in | Checked by |
|---|---|---|---|
| HTTP API (below) | `crates/senet-cli/src/server.rs` | `web/portable/local-api.js` (the portable page) | `tools/check_local_api.mjs` |
| C ABI | `crates/senet-ffi/include/senet.h` | `crates/senet-ffi/src/lib.rs`, `python/senet/_lib.py` | `python/tests/test_abi.py` |
| Python API | `python/senet/__init__.py` (its docstrings) | | `python/tests/test_api.py` |
| File formats | [FORMATS.md](FORMATS.md) | the Rust and Python readers and writers | `python/tests/test_manifest.py`, `check_all` |

## HTTP API

`senet serve` serves the web app (`web/`, or `--web`) and this API on `127.0.0.1` (port
8080, or `--port`). It answers only requests whose `Host` is `127.0.0.1` or `localhost` at
its port, so a web page cannot reach it through a domain that resolves to the machine (DNS
rebinding), and whose `Origin`, if they have one, is `http://127.0.0.1` or
`http://localhost` at its port, so a page of another site cannot use it either (403
otherwise). A POST must send its body as `Content-Type: application/json`, which no form
can. A request body may have up to 64 KiB and must arrive within 10 s; up to 64 connections
are served at once, and more wait. Each connection carries one request.

Boards are given in absolute colours: `white` and `black` list the squares (1 to 30,
never 27) that each side's pieces occupy, and `turn` (`"white"` or `"black"`) is the side
to throw. A position must be a game in progress: 1 to 5 pieces per side, on distinct
squares. A request with a field not listed below is refused.

### `GET /api/info`

What the server can do:

```json
{"ruleset": "kendall5",
 "database": {"complete": true, "path": "db/kendall5"},
 "network": true,
 "bots": ["perfect", "net:1", "net", "expectimax:3", "expectimax:2", "greedy", "random"],
 "start": {"white": [1, 3, 5, 7, 9], "black": [2, 4, 6, 8, 10], "turn": "white"},
 "throw_probs": {"1": 0.25, "2": 0.375, "3": 0.25, "4": 0.0625, "5": 0.0625},
 "extra_throws": [1, 4, 5]}
```

`database` is `null` without a database. The server loads `db/kendall5` and
`models/senet_net.bin` when they are there, and runs without what does not load; `--db`
and `--net` name others, which must load. `bots` lists, strongest first, the bots that
`/api/ai` plays here as themselves: `perfect` with the complete database, `net:1` and `net`
with a network, and always `expectimax:3`, `expectimax:2`, `greedy` and `random`.

### `POST /api/analyze`

Request: `{"white", "black", "turn", "throw"?, "eval"?}`. `eval` is `"auto"` (the default),
`"perfect"`, `"net"` or `"heuristic"`: the evaluator to use if it is available.

For the opening position and a throw of 2, which has one legal move, with the database:

```json
{"source": "perfect", "white_win_prob": 0.5025043487548828,
 "throw": 2,
 "moves": [{"from": 9, "to": 11, "kind": "move", "dir": "fwd",
            "white": [1, 3, 5, 7, 11], "black": [2, 4, 6, 8, 10],
            "next_turn": "black", "winner": null, "win_prob": 0.4870296120643616, "best": true}]}
```

* `source` is the evaluator that answered. `auto` takes the database if it covers the
  position, else the network if one is loaded, else the heuristic; `perfect` and `net` fall
  back to the heuristic when their evaluator is not available. Only `perfect` values are
  exact.
* `white_win_prob` is White's win probability in the position.
* With `throw` (1 to 5): `moves` lists every legal move, in the engine's order (by `from`,
  then `to`). `to` is where the piece comes to rest: 31 when borne off, and for a move onto
  the House of Water, the square it is sent back to. `kind` is `"move"`, `"swap"`, `"off"` or
  `"water"`; `dir` is `"fwd"` or `"back"`. `white` and `black` are the board after the
  move; `next_turn` is who throws next (`null` once the game is won) and `winner` the
  winner (or `null`). `win_prob` is the mover's win probability after the move, as the
  evaluator values the position it leads to, and `best` marks every move within 1e-9 of
  the best. With no legal move, `moves` is empty and `pass_to` names the side that throws
  next.

### `POST /api/ai`

Request: `{"white", "black", "turn", "throw", "bot"?, "seed"?}`. `bot` is one of the
`bots` of `/api/info`, or `"perfect"` (the default), which is always accepted; `seed`
(default 0) seeds the random bot.

```json
{"choice": 0, "bot": "perfect", "moves": [ ... ]}
```

`choice` indexes `moves`, which has the fields of `/api/analyze` without `best`; `win_prob`
is the bot's value of each move, or `null` for the random bot. With no legal move, `choice`
is `null` and `moves` empty. `bot` is the bot that chose: without the complete database,
`perfect` is played by `net:1` if a network is loaded, else by `expectimax:3`.

### Errors

The API answers a request it cannot serve with `{"error": "message"}` and the status:

| Status | When | Message |
|---|---|---|
| 400 | a body that is not the request's JSON, such as one with an unknown field | begins `bad request: ` |
| 400 | an invalid square or position, a finished game, a throw outside 1 to 5, an unknown bot, a `net` bot without a network | says which |
| 404 | an unknown endpoint under `/api/` | `unknown endpoint /api/...` |
| 405 | the wrong method, with an `Allow` header | `use POST for /api/analyze` |
| 415 | a POST whose `Content-Type` is not `application/json` | `send the body as Content-Type: application/json` |

The HTTP layer refuses some requests before they reach the API, with a plain-text body:
403 for a `Host` or an `Origin` other than this server, 408 for a request not received
within 10 s, 413 for a body over 64 KiB, 431 for a request head over 16 KiB, 400 for a
malformed request (the target must be a path, or an absolute `http://` URL), 417, 501 and
505 for what it does not support (an `Expect` other than `100-continue`, a
`Transfer-Encoding`, an HTTP version other than 1.x), and 500 if the server fails. After its
reply, the server reads what the client still sends for up to 1 s in all, then closes the
connection.

### The portable page

The portable page (`tools/build_portable.py`) answers the same requests in the browser,
with the network and no database: `web/portable/local-api.js` is a function of the path
and the request body, with no HTTP between them. Its replies have the server's keys and
types, and its failures are exceptions with the server's messages (for a malformed body,
only the `bad request: ` prefix is the same, and for an unknown field all but the position
serde appends). Otherwise it differs from the server in:

* `/api/info`: `"database": null`, `"network": true`, the bots `net:1`, `net` and `random`,
  and two more keys for the page header, `label` and `tagline`.
* `/api/analyze`: the network answers whatever `eval` asks for (`"source": "net"`), and
  values moves with a 1-throw search, as `net:1` does, where the server evaluates the
  positions they lead to.
* `/api/ai`: `perfect` is played by `net:1`. The random bot is seeded by `seed` but does
  not make the server's choices.

`tools/check_local_api.mjs` holds this API's reply shapes and failure messages. It checks
the portable implementation against them, and with `--server URL` a running server too;
`check_all` runs both checks.

## C ABI

[`crates/senet-ffi/include/senet.h`](../crates/senet-ffi/include/senet.h) declares it and
states its conventions: result codes, the per-thread message slot and its lifetime,
pointer and alignment rules, and what handles own and when they may be closed and freed.
`senet_abi_version()` returns `SENET_ABI_VERSION`; the Python package refuses a library of
another version.
