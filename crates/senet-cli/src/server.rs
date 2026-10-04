//! Local web server: serves the browser UI from `web/` and a small JSON API backed by
//! the perfect-play database (or the distilled network / heuristic as fallbacks). The API
//! is documented in docs/API.md. web/portable/local-api.js implements the same API in the
//! browser; tools/check_local_api.mjs checks that the two agree.
//!
//! The server listens on 127.0.0.1 only and answers only requests addressed to that host
//! (guarding against DNS rebinding) and, if they come from a web page, from its own pages.

use crate::http::{self, Limits, Request, Response};
use clap::Args;
use senet_core::board::{Bits, Pos, Rules, THROW_PROBS};
use senet_core::bots::{Bot, BotContext, BotError, RandomBot, Search, best_index, search_for};
use senet_core::db::Db;
use senet_core::eval::{Evaluator, Heuristic};
use senet_core::movegen::{Move, Outcome, legal_moves};
use senet_core::net::Net;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::net::TcpListener;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// The bots the web app offers, strongest first; `/api/ai` accepts only these.
const BOTS: &[&str] = &["perfect", "net:1", "net", "expectimax:3", "expectimax:2", "greedy", "random"];
/// Request bodies of up to 64 KiB, sent within 10 s, on up to 64 connections at once.
const LIMITS: Limits = Limits { body: 64 * 1024, timeout: Duration::from_secs(10), connections: 64 };
/// `/api/analyze` flags every move whose value is within this of the best as best. A
/// display threshold only: the bots break ties with `bots::TIE_TOLERANCE`, which stays
/// fixed so that matches replay identically.
const BEST_TOLERANCE: f64 = 1e-9;
const DEFAULT_DB: &str = "db/kendall5";
const DEFAULT_NET: &str = "models/senet_net.bin";

#[derive(Args)]
pub struct ServeArgs {
    /// Database directory [default: db/kendall5, if it loads]
    #[arg(long)]
    db: Option<PathBuf>,
    /// Network file [default: models/senet_net.bin, if it loads]
    #[arg(long)]
    net: Option<PathBuf>,
    /// Directory of the web app
    #[arg(long, default_value = "web")]
    web: PathBuf,
    /// Port to listen on (0: any free port; the address is printed)
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

struct State {
    rules: Rules,
    ctx: BotContext,
    web: PathBuf,
    /// The port the server listens on.
    port: u16,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Color {
    White,
    Black,
}

impl Color {
    fn other(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    /// The (white, black) masks of a position seen by a mover of this colour.
    fn absolute(self, pos: Pos) -> (u32, u32) {
        match self {
            Color::White => (pos.me, pos.opp),
            Color::Black => (pos.opp, pos.me),
        }
    }
}

/// Which evaluator `/api/analyze` should use; `auto` picks the best one available.
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum EvalChoice {
    #[default]
    Auto,
    Perfect,
    Net,
    Heuristic,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalyzeRequest {
    white: Vec<u32>,
    black: Vec<u32>,
    turn: Color,
    throw: Option<u8>,
    #[serde(default)]
    eval: EvalChoice,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AiRequest {
    white: Vec<u32>,
    black: Vec<u32>,
    turn: Color,
    throw: u8,
    bot: Option<String>,
    #[serde(default)]
    seed: u64,
}

#[derive(Serialize)]
struct MoveReply {
    from: u8,
    to: u8,
    kind: &'static str,
    dir: &'static str,
    white: Vec<u32>,
    black: Vec<u32>,
    next_turn: Option<Color>,
    winner: Option<Color>,
    /// The mover's win probability after the move (`None` for a bot without an evaluator).
    win_prob: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best: Option<bool>,
}

#[derive(Serialize)]
struct AnalyzeReply {
    source: &'static str,
    white_win_prob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    throw: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moves: Option<Vec<MoveReply>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pass_to: Option<Color>,
}

#[derive(Serialize)]
struct AiReply {
    choice: Option<usize>,
    moves: Vec<MoveReply>,
    bot: String,
}

#[derive(Serialize)]
struct Board {
    white: Vec<u32>,
    black: Vec<u32>,
    turn: Color,
}

#[derive(Serialize)]
struct DbInfo {
    complete: bool,
    path: String,
}

#[derive(Serialize)]
struct Info {
    ruleset: &'static str,
    database: Option<DbInfo>,
    network: bool,
    /// The offered bots that play as themselves here.
    bots: Vec<&'static str>,
    start: Board,
    throw_probs: BTreeMap<String, f64>,
    extra_throws: Vec<u8>,
}

/// An API failure: HTTP status and message.
#[derive(Debug)]
struct ApiError(u16, String);

impl From<String> for ApiError {
    fn from(msg: String) -> ApiError {
        ApiError(400, msg)
    }
}

impl From<&str> for ApiError {
    fn from(msg: &str) -> ApiError {
        ApiError(400, msg.to_string())
    }
}

impl From<BotError> for ApiError {
    fn from(e: BotError) -> ApiError {
        ApiError(400, e.to_string())
    }
}

fn squares(mask: u32) -> Vec<u32> {
    Bits(mask).collect()
}

fn check_throw(t: u8) -> Result<u8, ApiError> {
    if (1..=5).contains(&t) { Ok(t) } else { Err("throw must be 1..5".into()) }
}

/// The position from the point of view of `turn`, which must be a game in progress.
fn mover_view(white: &[u32], black: &[u32], turn: Color) -> Result<Pos, ApiError> {
    let pos = match turn {
        Color::White => Pos::from_squares(white, black)?,
        Color::Black => Pos::from_squares(black, white)?,
    };
    if pos.is_over() {
        return Err("the game is already over".into());
    }
    Ok(pos)
}

/// The evaluator to use for `pos` and its label for the `source` field. The database is
/// used if it covers `pos`, and so (as `Db::open` ensures) every position its moves lead to.
fn evaluator(st: &State, pos: Pos, choice: EvalChoice) -> (Arc<dyn Evaluator>, &'static str) {
    let db = st.ctx.db.as_ref().filter(|db| db.lookup(pos).is_some());
    if let (EvalChoice::Auto | EvalChoice::Perfect, Some(db)) = (choice, db) {
        return (db.clone(), "perfect");
    }
    if let (EvalChoice::Auto | EvalChoice::Net, Some(net)) = (choice, &st.ctx.net) {
        return (net.clone(), "net");
    }
    (Arc::new(Heuristic), "heuristic")
}

fn move_reply(rules: &Rules, m: &Move, t: u8, mover: Color, win_prob: Option<f64>, best: Option<bool>) -> MoveReply {
    let (white, black) = mover.absolute(m.after);
    let (next_turn, winner) = match m.outcome(rules, t) {
        Outcome::Won => (None, Some(mover)),
        Outcome::ThrowAgain(_) => (Some(mover), None),
        Outcome::OpponentThrows(_) => (Some(mover.other()), None),
    };
    MoveReply {
        from: m.from,
        to: m.to,
        kind: m.kind.name(),
        dir: m.direction_name(),
        white: squares(white),
        black: squares(black),
        next_turn,
        winner,
        win_prob,
        best,
    }
}

fn analyze(st: &State, req: AnalyzeRequest) -> Result<AnalyzeReply, ApiError> {
    let pos = mover_view(&req.white, &req.black, req.turn)?;
    let (eval, source) = evaluator(st, pos, req.eval);
    let v_mover = eval.value(pos);
    let mut reply = AnalyzeReply {
        source,
        white_win_prob: if req.turn == Color::White { v_mover } else { 1.0 - v_mover },
        throw: None,
        moves: None,
        pass_to: None,
    };
    if let Some(t) = req.throw {
        let t = check_throw(t)?;
        let moves = legal_moves(&st.rules, pos, t);
        let search = Search { eval, depth: 0 };
        let values = search.move_values(&st.rules, t, &moves);
        let best = values.iter().copied().fold(f64::MIN, f64::max);
        reply.throw = Some(t);
        reply.moves = Some(
            moves
                .iter()
                .zip(&values)
                .map(|(m, &v)| move_reply(&st.rules, m, t, req.turn, Some(v), Some(best - v < BEST_TOLERANCE)))
                .collect(),
        );
        reply.pass_to = moves.is_empty().then(|| req.turn.other());
    }
    Ok(reply)
}

fn ai(st: &State, req: AiRequest) -> Result<AiReply, ApiError> {
    let pos = mover_view(&req.white, &req.black, req.turn)?;
    let t = check_throw(req.throw)?;
    let requested = req.bot.as_deref().unwrap_or("perfect");
    let Some(&spec) = BOTS.iter().find(|&&b| b == requested) else {
        return Err(format!("unknown bot '{requested}' (available: {})", BOTS.join(", ")).into());
    };
    // Perfect play needs the complete database; without it the strongest available bot
    // stands in.
    let complete_db = st.ctx.complete_db().is_ok();
    let spec = match spec {
        "perfect" if !complete_db && st.ctx.net.is_some() => "net:1",
        "perfect" if !complete_db => "expectimax:3",
        spec => spec,
    };
    let moves = legal_moves(&st.rules, pos, t);
    if moves.is_empty() {
        return Ok(AiReply { choice: None, moves: vec![], bot: spec.to_string() });
    }
    // A bot with an evaluator plays the best of the values it reports; `random` has none.
    let (choice, values): (usize, Vec<Option<f64>>) = match search_for(spec, &st.ctx)? {
        Some(search) => {
            let values = search.move_values(&st.rules, t, &moves);
            (best_index(&values), values.into_iter().map(Some).collect())
        }
        None => (RandomBot::new(req.seed).choose(&st.rules, pos, t, &moves), vec![None; moves.len()]),
    };
    let moves = moves.iter().zip(values).map(|(m, v)| move_reply(&st.rules, m, t, req.turn, v, None)).collect();
    Ok(AiReply { choice: Some(choice), moves, bot: spec.to_string() })
}

fn info(st: &State) -> Info {
    let start = st.rules.start();
    Info {
        ruleset: "kendall5",
        database: st.ctx.db.as_ref().map(|d| DbInfo { complete: d.is_complete(), path: d.dir().display().to_string() }),
        network: st.ctx.net.is_some(),
        // `perfect` needs the complete database and the `net` bots a network.
        bots: BOTS.iter().copied().filter(|bot| search_for(bot, &st.ctx).is_ok()).collect(),
        start: Board { white: squares(start.me), black: squares(start.opp), turn: Color::White },
        throw_probs: (1..=5usize).map(|t| (t.to_string(), THROW_PROBS[t])).collect(),
        extra_throws: (1..=5).filter(|&t| st.rules.extra_throw(t)).collect(),
    }
}

fn json_reply(status: u16, body: &impl Serialize) -> Response {
    Response::new(status, "application/json", serde_json::to_vec(body).expect("serializable reply"))
}

fn error_reply(e: ApiError) -> Response {
    json_reply(e.0, &serde_json::json!({ "error": e.1 }))
}

/// Whether a request's `Host` header names this server, so that a web page whose domain
/// resolves to 127.0.0.1 cannot use the API (DNS rebinding).
fn host_allowed(host: Option<&str>, port: u16) -> bool {
    host.is_some_and(|host| {
        let name = host.strip_suffix(&format!(":{port}")).unwrap_or(host);
        name == "127.0.0.1" || name.eq_ignore_ascii_case("localhost")
    })
}

/// Whether a request's `Origin` header, if it has one, names this server. A page of another
/// site can send any server the requests a form can (such as a POST of plain text), though
/// it cannot read the replies; the server refuses them.
fn origin_allowed(origin: Option<&str>, port: u16) -> bool {
    origin.is_none_or(|origin| {
        ["127.0.0.1", "localhost"]
            .iter()
            .any(|name| origin == format!("http://{name}:{port}") || (port == 80 && origin == format!("http://{name}")))
    })
}

/// Whether a `Content-Type` header value is JSON's, which no form can send.
fn is_json(content_type: Option<&str>) -> bool {
    content_type
        .is_some_and(|t| t.split(';').next().unwrap_or_default().trim().eq_ignore_ascii_case("application/json"))
}

fn read_json<T: DeserializeOwned>(body: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError(400, format!("bad request: {e}")))
}

/// Answers an API request, to `/api/{endpoint}`.
fn api(st: &State, req: &Request, endpoint: &str) -> Response {
    let expected = match endpoint {
        "info" => "GET",
        "analyze" | "ai" => "POST",
        _ => return error_reply(ApiError(404, format!("unknown endpoint /api/{endpoint}"))),
    };
    if req.method != expected {
        return error_reply(ApiError(405, format!("use {expected} for /api/{endpoint}")))
            .with_header("Allow", expected);
    }
    if expected == "POST" && !is_json(req.header("content-type")) {
        return error_reply(ApiError(415, "send the body as Content-Type: application/json".into()));
    }
    let reply = match endpoint {
        "info" => Ok(json_reply(200, &info(st))),
        "analyze" => read_json(&req.body).and_then(|r| analyze(st, r)).map(|r| json_reply(200, &r)),
        _ => read_json(&req.body).and_then(|r| ai(st, r)).map(|r| json_reply(200, &r)),
    };
    reply.unwrap_or_else(error_reply)
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "json" => "application/json",
        "png" => "image/png",
        _ => "application/octet-stream",
    }
}

/// The contents of `path` if it is a regular file (on Windows, `nul` opens in any directory).
fn read_regular_file(path: &Path) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(io::ErrorKind::NotFound.into());
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// A file below the web directory. Only plain path components are accepted: no `..`,
/// roots, drive prefixes or backslash tricks, and no `:`, with which Windows names a
/// stream of a file (`index.html::$DATA`).
fn static_file(st: &State, path: &str) -> Response {
    let rel = Path::new(match path.trim_start_matches('/') {
        "" => "index.html",
        rel => rel,
    });
    let plain = |c: Component| matches!(c, Component::Normal(name) if !name.to_string_lossy().contains(':'));
    if !rel.components().all(plain) {
        return Response::text(400, "bad path");
    }
    match read_regular_file(&st.web.join(rel)) {
        Ok(bytes) => Response::new(200, content_type(rel), bytes).with_header("Cache-Control", "no-cache"),
        Err(_) => Response::text(404, "not found"),
    }
}

/// Answers a request: the API below /api/, the web app's files elsewhere.
fn handle(st: &State, req: &Request) -> Response {
    if !host_allowed(req.host.as_deref(), st.port) {
        return Response::text(403, "forbidden host");
    }
    if !origin_allowed(req.header("origin"), st.port) {
        return Response::text(403, "forbidden origin");
    }
    match req.path.strip_prefix("/api/") {
        Some(endpoint) => api(st, req, endpoint),
        None if req.method == "GET" => static_file(st, &req.path),
        None => Response::text(405, "use GET for the web app's files").with_header("Allow", "GET"),
    }
}

/// Loads a resource from an explicit path (failing if it does not load) or else from its
/// default path (running without it if that does not load).
fn resource<T>(
    explicit: Option<&Path>,
    default: &str,
    what: &str,
    load: impl Fn(&Path) -> Result<T, String>,
) -> Result<Option<T>, String> {
    if let Some(path) = explicit {
        return load(path).map(Some).map_err(|e| format!("loading {what} {}: {e}", path.display()));
    }
    match load(Path::new(default)) {
        Ok(x) => Ok(Some(x)),
        Err(e) => {
            eprintln!("{what} not loaded ({default}: {e})");
            Ok(None)
        }
    }
}

pub fn serve(a: ServeArgs) -> Result<(), String> {
    if !a.web.join("index.html").is_file() {
        return Err(format!(
            "{}: the web app is not there (no index.html); run `senet serve` in the repository's directory, or \
             give the web app's directory with --web",
            a.web.display()
        ));
    }
    let db = resource(a.db.as_deref(), DEFAULT_DB, "database", |p| Db::open(p).map_err(|e| e.to_string()))?;
    let net = resource(a.net.as_deref(), DEFAULT_NET, "network", |p| Net::load(p).map_err(|e| e.to_string()))?;
    let fallback = "fall back to the network or the heuristic";
    match &db {
        Some(db) if db.is_complete() => eprintln!("database: {} (complete)", db.dir().display()),
        Some(db) => {
            eprintln!("database: {} (partial: hints outside it and the perfect bot {fallback})", db.dir().display())
        }
        None => eprintln!("no database: hints and the perfect bot {fallback}"),
    }
    let ctx = BotContext { db: db.map(Arc::new), net: net.map(Arc::new) };
    let listener =
        TcpListener::bind(("127.0.0.1", a.port)).map_err(|e| format!("listening on 127.0.0.1:{}: {e}", a.port))?;
    let port = listener.local_addr().map_err(|e| format!("listening on 127.0.0.1:{}: {e}", a.port))?.port();
    let st = State { rules: Rules::KENDALL5, ctx, web: a.web, port };
    eprintln!("Senet server on http://127.0.0.1:{port}/");
    http::run(listener, LIMITS, move |req| handle(&st, &req))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> State {
        let web = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web");
        State { rules: Rules::KENDALL5, ctx: BotContext::default(), web, port: 8080 }
    }

    fn analyze_json(st: &State, body: &str) -> Result<AnalyzeReply, ApiError> {
        analyze(st, serde_json::from_str(body).map_err(|e| ApiError(400, e.to_string()))?)
    }

    #[test]
    fn analyze_validates_and_reports_moves() {
        let st = state();
        let r = analyze_json(&st, r#"{"white":[1,3,5,7,9],"black":[2,4,6,8,10],"turn":"white","throw":1}"#).unwrap();
        assert_eq!(r.source, "heuristic");
        let moves = r.moves.unwrap();
        assert!(!moves.is_empty() && moves.iter().any(|m| m.best == Some(true)));
        assert!(moves.iter().all(|m| m.next_turn == Some(Color::White)), "a 1 grants another throw");
        for bad in [
            r#"{"white":[1],"black":[1],"turn":"white"}"#,
            r#"{"white":[27],"black":[1],"turn":"white"}"#,
            r#"{"white":[1,1],"black":[2],"turn":"white"}"#,
            r#"{"white":[1],"black":[],"turn":"white"}"#,
            r#"{"white":[1],"black":[2],"turn":"red"}"#,
            r#"{"white":[1],"black":[2],"turn":"white","throw":6}"#,
            r#"{"white":[1],"black":[2],"turn":"white","throw":"2"}"#,
            r#"{"white":[1],"black":[2],"turn":"white","eval":"oracle"}"#,
        ] {
            assert!(analyze_json(&st, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn unknown_fields_are_refused() {
        let st = state();
        for (endpoint, body) in [
            ("analyze", r#"{"white":[1],"black":[2],"turn":"white","thorw":2}"#),
            ("ai", r#"{"white":[1],"black":[2],"turn":"white","throw":2,"thorw":2}"#),
        ] {
            let r = handle(&st, &request("POST", &format!("/api/{endpoint}"), JSON, body.as_bytes()));
            let error = String::from_utf8(r.body).unwrap();
            assert_eq!(r.status, 400, "{error}");
            assert!(error.contains("bad request: unknown field `thorw`, expected one of `white`, `black`"), "{error}");
        }
    }

    #[test]
    fn ai_accepts_only_offered_bots() {
        let st = state();
        let req = |bot: &str| AiRequest {
            white: vec![1, 3],
            black: vec![2, 4],
            turn: Color::Black,
            throw: 3,
            bot: Some(bot.to_string()),
            seed: 1,
        };
        let r = ai(&st, req("greedy")).unwrap();
        assert_eq!(r.moves.len(), 2);
        assert!(r.choice.unwrap() < 2 && r.moves.iter().all(|m| m.win_prob.is_some() && m.best.is_none()));
        let r = ai(&st, req("random")).unwrap();
        assert!(r.moves.iter().all(|m| m.win_prob.is_none()));
        assert_eq!(ai(&st, req("perfect")).unwrap().bot, "expectimax:3", "no database or network loaded");
        assert!(ai(&st, req("net:1")).is_err(), "no network loaded");
        assert!(ai(&st, req("expectimax:4")).is_err());
    }

    #[test]
    fn info_lists_the_bots_that_play_as_themselves() {
        let st = state();
        assert_eq!(info(&st).bots, ["expectimax:3", "expectimax:2", "greedy", "random"]);
    }

    #[test]
    fn only_requests_addressed_to_this_server_are_served() {
        for ok in ["127.0.0.1:8080", "localhost:8080", "LOCALHOST:8080", "127.0.0.1", "localhost"] {
            assert!(host_allowed(Some(ok), 8080), "{ok}");
        }
        for bad in ["127.0.0.1:9999", "evil.com", "evil.com:8080", "localhost.evil.com:8080", "127.0.0.1:8080:8080", ""]
        {
            assert!(!host_allowed(Some(bad), 8080), "{bad}");
        }
        assert!(!host_allowed(None, 8080), "a request without a Host header");
    }

    /// The headers of a request from the web app.
    const JSON: &[(&str, &str)] = &[("content-type", "application/json"), ("origin", "http://127.0.0.1:8080")];

    fn request(method: &str, path: &str, headers: &[(&str, &str)], body: &[u8]) -> Request {
        Request {
            method: method.into(),
            path: path.into(),
            host: Some("127.0.0.1:8080".into()),
            headers: headers.iter().map(|&(n, v)| (n.into(), v.into())).collect(),
            body: body.into(),
        }
    }

    #[test]
    fn routing() {
        let st = state();
        let status = |method: &str, path: &str| handle(&st, &request(method, path, JSON, b"")).status;
        assert_eq!(status("GET", "/api/info"), 200);
        assert_eq!(status("GET", "/api/nope"), 404);
        assert_eq!(status("GET", "/api/ai"), 405);
        assert_eq!(status("POST", "/api/info"), 405);
        assert_eq!(status("HEAD", "/api/info"), 405);
        assert_eq!(status("POST", "/api/analyze"), 400);
        assert_eq!(status("GET", "/app.js"), 200);
        assert_eq!(status("POST", "/app.js"), 405);
        let elsewhere = Request { host: Some("evil.com".into()), ..request("GET", "/api/info", &[], b"") };
        assert_eq!(handle(&st, &elsewhere).status, 403);
    }

    #[test]
    fn requests_from_other_sites_are_refused() {
        let st = state();
        let board = br#"{"white":[1,3],"black":[2,4],"turn":"white"}"#;
        let status = |headers: &[(&str, &str)]| handle(&st, &request("POST", "/api/analyze", headers, board)).status;
        assert_eq!(status(JSON), 200);
        assert_eq!(status(&[("content-type", "application/json; charset=utf-8")]), 200, "no Origin: not a web page");
        assert_eq!(status(&[("content-type", "Application/JSON"), ("origin", "http://localhost:8080")]), 200);
        // What a form or a "simple" cross-site fetch can send.
        assert_eq!(status(&[("content-type", "text/plain"), ("origin", "http://127.0.0.1:8080")]), 415);
        assert_eq!(status(&[("origin", "http://127.0.0.1:8080")]), 415);
        for origin in
            ["https://evil.com", "null", "http://127.0.0.1:9999", "http://127.0.0.1", "https://127.0.0.1:8080"]
        {
            assert_eq!(status(&[("content-type", "application/json"), ("origin", origin)]), 403, "{origin}");
        }
        assert!(origin_allowed(Some("http://localhost"), 80) && !origin_allowed(Some("http://localhost"), 8080));
    }

    #[test]
    fn serves_on_the_port_the_system_assigns() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let st = State { port, ..state() };
        std::thread::spawn(move || http::run(listener, LIMITS, move |req| handle(&st, &req)));
        let get =
            |host: &str| http::exchange(port, format!("GET /api/info HTTP/1.1\r\nHost: {host}\r\n\r\n").as_bytes());
        assert!(get(&format!("127.0.0.1:{port}")).starts_with("HTTP/1.1 200 OK"));
        assert!(get("127.0.0.1:0").starts_with("HTTP/1.1 403"));
        // A body of the largest size accepted.
        let board = r#"{"white":[1,3,5,7,9],"black":[2,4,6,8,10],"turn":"white"}"#;
        let body = board.to_string() + &" ".repeat(LIMITS.body - board.len());
        let post = format!(
            "POST /api/analyze HTTP/1.1\r\nHost: localhost:{port}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let reply = http::exchange(port, post.as_bytes());
        assert!(reply.starts_with("HTTP/1.1 200 OK") && reply.contains(r#""white_win_prob""#), "{reply}");
    }

    #[test]
    fn hints_are_perfect_inside_a_partial_database() {
        use senet_core::db::layer_path;
        use senet_core::solver::{SolveConfig, solve_all};
        let dir = std::env::temp_dir().join(format!("senet_server_db_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        solve_all(&Rules::KENDALL5, &dir, &SolveConfig { max_sum: 3, ..SolveConfig::DEFAULT }).unwrap();
        let st = State { ctx: BotContext::load(Some(&dir), None).unwrap(), ..state() };
        let db = st.ctx.db.clone().unwrap();
        // Both moves lead into layer (1, 2), with Black to throw.
        let r = analyze_json(&st, r#"{"white":[20,24],"black":[10],"turn":"white","throw":2}"#).unwrap();
        assert_eq!(r.source, "perfect");
        for m in r.moves.unwrap() {
            let after = Pos::from_squares(&m.black, &m.white).unwrap();
            assert_eq!(m.win_prob, Some(1.0 - db.lookup(after).unwrap()));
        }
        let r = analyze_json(&st, r#"{"white":[1,3],"black":[2,4],"turn":"white","throw":1}"#).unwrap();
        assert_eq!(r.source, "heuristic", "2 v 2 is outside the database");
        drop((st, db));
        // Without layer (1, 2) the database would cover (2, 1) but not its moves.
        std::fs::remove_file(layer_path(&dir, 1, 2)).unwrap();
        assert!(BotContext::load(Some(&dir), None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn static_paths_stay_inside_the_web_directory() {
        let st = state();
        assert_eq!(static_file(&st, "/").status, 200);
        assert_eq!(static_file(&st, "/app.js").status, 200);
        assert_eq!(static_file(&st, "/../Cargo.toml").status, 400);
        for bad in ["/a/../../Cargo.toml", "/C:/Windows/win.ini", r"/..\Cargo.toml", "/missing.js"] {
            assert_ne!(static_file(&st, bad).status, 200, "{bad}");
        }
        // Only regular files: not a directory, a device or a named stream of a file.
        for bad in ["/portable", "/nul", "/NUL", "/nul.js", "/index.html::$DATA", "/index.html:x"] {
            assert_ne!(static_file(&st, bad).status, 200, "{bad}");
        }
    }

    #[test]
    fn serving_needs_the_web_app() {
        let args = ServeArgs { db: None, net: None, web: PathBuf::from("no/such/web"), port: 0 };
        let e = serve(args).unwrap_err();
        assert!(e.contains("no/such/web") && e.contains("--web"), "{e}");
    }
}
