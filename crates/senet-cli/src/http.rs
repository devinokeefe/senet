//! A small HTTP/1.1 server for the local web app. Each connection carries one request,
//! answered on a thread of its own and then closed. A request must arrive whole within a
//! time limit and its declared body must fit a size limit before any of it is read, so a
//! slow or hostile client can hold a connection only briefly and cannot make the server
//! allocate much. At most a limited number of connections are served at once; further
//! ones wait in the listen queue.

use std::fmt::Write as _;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Largest request head: the request line and the headers.
const MAX_HEAD: usize = 16 * 1024;
/// How long, in total, a connection stays open after its reply to take in what the client
/// still sends (closing it with unread data would reset it, and could discard the reply).
/// It takes in at most the size of a whole request, `MAX_HEAD + Limits::body` bytes.
const LINGER: Duration = Duration::from_secs(1);

#[derive(Clone, Copy)]
pub struct Limits {
    /// Largest request body.
    pub body: usize,
    /// Time a client has to send its whole request, and to take each write of the reply.
    pub timeout: Duration,
    /// Connections served at once.
    pub connections: usize,
}

pub struct Request {
    pub method: String,
    /// The request target without its query.
    pub path: String,
    /// The host the request is addressed to: that of an absolute-form target, else the
    /// `Host` header.
    pub host: Option<String>,
    /// The headers, with lowercase names.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    /// The value of the first header named `name` (lowercase).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }
}

pub struct Response {
    pub status: u16,
    pub headers: Vec<(&'static str, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16, content_type: &str, body: impl Into<Vec<u8>>) -> Response {
        Response { status, headers: vec![("Content-Type", content_type.to_string())], body: body.into() }
    }

    pub fn text(status: u16, text: &str) -> Response {
        Response::new(status, "text/plain; charset=utf-8", text)
    }

    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Response {
        self.headers.push((name, value.into()));
        self
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        417 => "Expectation Failed",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        505 => "HTTP Version Not Supported",
        _ => "",
    }
}

/// Reads from a connection until a deadline: each read waits at most the time left.
struct Deadline<'a> {
    stream: &'a TcpStream,
    until: Instant,
}

impl Read for Deadline<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = self.until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(io::ErrorKind::TimedOut.into());
        }
        self.stream.set_read_timeout(Some(left))?;
        self.stream.read(buf)
    }
}

/// The reply to a request that could not be read whole.
fn read_failure(e: io::Error) -> Response {
    match e.kind() {
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => Response::text(408, "request timed out"),
        _ => Response::text(400, "incomplete request"),
    }
}

/// Reads the request head up to and including its empty line.
fn read_head(reader: &mut impl BufRead) -> Result<String, Response> {
    let mut head = Vec::new();
    let mut limited = reader.take(MAX_HEAD as u64);
    loop {
        let start = head.len();
        limited.read_until(b'\n', &mut head).map_err(read_failure)?;
        let line = &head[start..];
        if !line.ends_with(b"\n") {
            return Err(if limited.limit() == 0 {
                Response::text(431, "request head too large")
            } else {
                Response::text(400, "incomplete request")
            });
        }
        if line == b"\r\n" || line == b"\n" {
            break;
        }
    }
    String::from_utf8(head).map_err(|_| Response::text(400, "request head is not UTF-8"))
}

/// The body length a request declares (0 if it declares none).
fn content_length(headers: &[(String, String)]) -> Result<u64, Response> {
    let mut values = headers.iter().filter(|(name, _)| name == "content-length").map(|(_, v)| v.as_str());
    let Some(first) = values.next() else { return Ok(0) };
    if values.any(|v| v != first) {
        return Err(Response::text(400, "conflicting Content-Length headers"));
    }
    if first.is_empty() || !first.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Response::text(400, "invalid Content-Length"));
    }
    Ok(first.parse().unwrap_or(u64::MAX)) // more digits than a u64 holds is too large anyway
}

/// Whether `s` is a token, as a method must be (RFC 9110, section 5.6.2).
fn is_token(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

/// The authority and the path of a request target: `/path?query` has no authority;
/// `http://host/path?query` (the absolute form) has one. `None` for any other form.
fn split_target(target: &str) -> Option<(Option<&str>, &str)> {
    if target.starts_with('/') {
        return Some((None, target));
    }
    if !target.get(..7)?.eq_ignore_ascii_case("http://") {
        return None;
    }
    let rest = &target[7..];
    let (authority, path) = rest.split_at(rest.find(['/', '?']).unwrap_or(rest.len()));
    Some((Some(authority), if path.starts_with('/') { path } else { "/" }))
}

/// Reads one request from `stream`. A request that is malformed, too large, too slow or
/// needs a feature this server lacks yields the reply to send instead.
fn read_request(stream: &TcpStream, limits: &Limits) -> Result<Request, Response> {
    let mut reader = BufReader::new(Deadline { stream, until: Instant::now() + limits.timeout });
    let head = read_head(&mut reader)?;
    let mut lines = head.lines();
    let request_line: Vec<&str> = lines.next().unwrap_or_default().split(' ').collect();
    let [method, target, version] = request_line[..] else {
        return Err(Response::text(400, "malformed request line"));
    };
    if !is_token(method) {
        return Err(Response::text(400, "malformed request line"));
    }
    let Some((authority, target)) = split_target(target) else {
        return Err(Response::text(400, "unsupported request target"));
    };
    if !version.starts_with("HTTP/1.") {
        return Err(Response::text(505, "HTTP/1.x only"));
    }
    let mut headers = Vec::new();
    for line in lines.take_while(|line| !line.is_empty()) {
        match line.split_once(':') {
            Some((name, value)) if !name.is_empty() && !name.contains(|c: char| c.is_ascii_whitespace()) => {
                headers.push((name.to_ascii_lowercase(), value.trim().to_string()));
            }
            _ => return Err(Response::text(400, "malformed header")),
        }
    }
    let header = |name: &str| headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str());
    if header("transfer-encoding").is_some() {
        return Err(Response::text(501, "Transfer-Encoding is not supported; send a Content-Length"));
    }
    let continue_expected = match header("expect") {
        None => false,
        Some(v) if v.eq_ignore_ascii_case("100-continue") => true,
        Some(_) => return Err(Response::text(417, "unsupported expectation")),
    };
    let len = content_length(&headers)?;
    if len > limits.body as u64 {
        return Err(Response::text(413, &format!("request body over {} bytes", limits.body)));
    }
    if continue_expected && len > 0 {
        let mut stream = stream;
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").map_err(read_failure)?;
    }
    let mut body = vec![0; len as usize];
    reader.read_exact(&mut body).map_err(read_failure)?;
    let host = authority.or(header("host")).map(str::to_string);
    Ok(Request {
        method: method.to_string(),
        path: target.split('?').next().unwrap_or_default().to_string(),
        host,
        headers,
        body,
    })
}

fn write_response(mut stream: &TcpStream, r: &Response) -> io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        r.status,
        reason(r.status),
        r.body.len()
    );
    for (name, value) in &r.headers {
        let _ = write!(head, "{name}: {value}\r\n");
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(&r.body)?;
    stream.flush()
}

/// Answers the one request of a connection, then closes it.
fn connection(stream: &TcpStream, limits: &Limits, handler: &(dyn Fn(Request) -> Response + Sync)) {
    let _ = stream.set_write_timeout(Some(limits.timeout));
    let response = match read_request(stream, limits) {
        // A panic is a bug; the panic hook reports it, and the client learns of it.
        Ok(request) => catch_unwind(AssertUnwindSafe(|| handler(request)))
            .unwrap_or_else(|_| Response::text(500, "internal error")),
        Err(response) => response,
    };
    if write_response(stream, &response).is_ok() && stream.shutdown(Shutdown::Write).is_ok() {
        let rest = Deadline { stream, until: Instant::now() + LINGER };
        let _ = io::copy(&mut rest.take((MAX_HEAD + limits.body) as u64), &mut io::sink());
    }
}

/// The number of connections being served.
struct Open {
    count: Mutex<usize>,
    freed: Condvar,
}

/// A connection being served, which gives back its place when dropped.
struct Slot(Arc<Open>);

impl Drop for Slot {
    fn drop(&mut self) {
        *self.0.count.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
        self.0.freed.notify_one();
    }
}

/// Serves `listener` forever, answering each request with `handler`.
pub fn run(listener: TcpListener, limits: Limits, handler: impl Fn(Request) -> Response + Send + Sync + 'static) -> ! {
    let handler = Arc::new(handler);
    let open = Arc::new(Open { count: Mutex::new(0), freed: Condvar::new() });
    loop {
        // At the limit, wait for a connection to end before accepting the next one.
        let count = open.count.lock().unwrap_or_else(PoisonError::into_inner);
        let mut count =
            open.freed.wait_while(count, |n| *n >= limits.connections).unwrap_or_else(PoisonError::into_inner);
        *count += 1;
        drop(count);
        let slot = Slot(open.clone());
        let stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(e) => {
                eprintln!("accepting a connection: {e}");
                std::thread::sleep(Duration::from_millis(100)); // in case the failure persists
                continue;
            }
        };
        let handler = handler.clone();
        let spawned = std::thread::Builder::new().name("http connection".into()).spawn(move || {
            let _slot = slot;
            connection(&stream, &limits, &*handler);
        });
        if let Err(e) = spawned {
            eprintln!("starting a thread for a connection: {e}");
        }
    }
}

/// Sends `raw` to the server on `port` and returns its whole reply.
#[cfg(test)]
pub fn exchange(port: u16, raw: &[u8]) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.write_all(raw).unwrap();
    let mut reply = String::new();
    stream.read_to_string(&mut reply).unwrap();
    reply
}

#[cfg(test)]
mod tests {
    use super::*;
    use senet_core::rng::Rng;

    const LIMITS: Limits = Limits { body: 16, timeout: Duration::from_millis(500), connections: 3 };
    /// For requests sent in pieces, which on a busy machine can take a while.
    const PATIENT: Limits = Limits { timeout: Duration::from_secs(10), ..LIMITS };

    /// A server that echoes the method, path and body of each request, and its host in an
    /// `X-Host` header (or panics).
    fn echo_server() -> u16 {
        echo_server_with(LIMITS)
    }

    fn echo_server_with(limits: Limits) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            run(listener, limits, |r| {
                assert!(r.path != "/panic", "a handler bug");
                let body = format!("{} {} {}", r.method, r.path, String::from_utf8_lossy(&r.body));
                Response::text(200, &body).with_header("X-Host", r.host.unwrap_or_default())
            })
        });
        port
    }

    fn status(reply: &str) -> u16 {
        reply.split(' ').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0)
    }

    #[test]
    fn answers_well_formed_requests() {
        let port = echo_server();
        let reply = exchange(port, b"POST /a/b?x=1 HTTP/1.1\r\nHost: h\r\nContent-Length: 5\r\n\r\nhello");
        assert!(reply.starts_with("HTTP/1.1 200 OK\r\n") && reply.ends_with("\r\n\r\nPOST /a/b hello"), "{reply}");
        assert!(reply.contains("Connection: close\r\n") && reply.contains("Content-Length: 15\r\n"), "{reply}");
        let reply = exchange(port, b"GET / HTTP/1.0\n\n");
        assert!(reply.ends_with("GET / "), "{reply}");
        // With `Expect: 100-continue`, the client waits for the go-ahead before the body.
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream.write_all(b"POST / HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\n").unwrap();
        let mut go = [0u8; 25];
        stream.read_exact(&mut go).unwrap();
        assert_eq!(&go, b"HTTP/1.1 100 Continue\r\n\r\n");
        stream.write_all(b"ok").unwrap();
        let mut reply = String::new();
        stream.read_to_string(&mut reply).unwrap();
        assert!(reply.ends_with("POST / ok"), "{reply}");
        assert_eq!(status(&exchange(port, b"GET /panic HTTP/1.1\r\n\r\n")), 500);
    }

    #[test]
    fn refuses_bad_requests_without_reading_their_bodies() {
        let port = echo_server();
        for (raw, expected) in [
            (&b"POST / HTTP/1.1\r\nContent-Length: 4611686018427387904\r\n\r\n"[..], 413),
            (b"POST / HTTP/1.1\r\nContent-Length: 99999999999999999999999\r\n\r\n", 413),
            (b"POST / HTTP/1.1\r\nContent-Length: 17\r\n\r\n", 413),
            (b"POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n", 400),
            (b"POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n", 400),
            (b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", 501),
            (b"POST / HTTP/1.1\r\nExpect: magic\r\n\r\n", 417),
            (b"GET /\r\n\r\n", 400),
            (b"GET / HTTP/2\r\n\r\n", 505),
            (b"GET / HTTP/1.1\r\nno colon\r\n\r\n", 400),
            (b"GET / HTTP/1.1\r\n folded: header\r\n\r\n", 400),
        ] {
            assert_eq!(status(&exchange(port, raw)), expected, "{}", String::from_utf8_lossy(raw));
        }
        let huge = [b"GET / HTTP/1.1\r\nX: ".as_slice(), &[b'x'; MAX_HEAD], b"\r\n\r\n"].concat();
        assert_eq!(status(&exchange(port, &huge)), 431);
    }

    #[test]
    fn request_lines() {
        let port = echo_server();
        // The absolute form, which names the host in the target.
        let reply = exchange(port, b"GET http://localhost:8080/a/b?x=1 HTTP/1.1\r\nHost: other\r\n\r\n");
        assert!(reply.contains("X-Host: localhost:8080\r\n") && reply.ends_with("GET /a/b "), "{reply}");
        let reply = exchange(port, b"GET HTTP://h?x HTTP/1.1\r\n\r\n");
        assert!(reply.contains("X-Host: h\r\n") && reply.ends_with("GET / "), "{reply}");
        let reply = exchange(port, b"GET / HTTP/1.1\r\nHost: h:1\r\n\r\n");
        assert!(reply.contains("X-Host: h:1\r\n"), "{reply}");
        for raw in [
            &b" / HTTP/1.1\r\n\r\n"[..],
            b"G(T / HTTP/1.1\r\n\r\n",
            b"GET a HTTP/1.1\r\n\r\n",
            b"OPTIONS * HTTP/1.1\r\n\r\n",
            b"GET https://h/ HTTP/1.1\r\n\r\n",
        ] {
            assert_eq!(status(&exchange(port, raw)), 400, "{}", String::from_utf8_lossy(raw));
        }
    }

    /// Sends `raw` to the server on `port` in pieces of 1..=`piece` bytes, pausing between
    /// them, then ends the request (a server still waiting for bytes sees the end) and
    /// returns the whole reply. The server may reply, and close, before it has all of it.
    fn exchange_in_pieces(port: u16, raw: &[u8], piece: u64, rng: &mut Rng) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream.set_nodelay(true).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut rest = raw;
        while !rest.is_empty() {
            let n = (1 + rng.below(piece) as usize).min(rest.len());
            if stream.write_all(&rest[..n]).is_err() {
                break;
            }
            rest = &rest[n..];
            if piece < raw.len() as u64 {
                pause(Duration::from_micros(200));
            }
        }
        let _ = stream.shutdown(Shutdown::Write);
        let mut reply = Vec::new();
        stream.read_to_end(&mut reply).expect("a whole reply before the connection closes");
        String::from_utf8(reply).unwrap()
    }

    /// Waits about `d`, yielding to other threads. A sleep would do, but Windows rounds it up
    /// to the system's timer tick, 15.6 ms unless a program asks for finer: then a request
    /// sent a byte at a time outlasts the server's timeout.
    fn pause(d: Duration) {
        let until = Instant::now() + d;
        while Instant::now() < until {
            std::thread::yield_now();
        }
    }

    /// The status of a well-formed reply, whose Content-Length is its body's length (after
    /// the go-ahead to an `Expect: 100-continue` request).
    fn checked_status(reply: &str) -> u16 {
        let reply = reply.strip_prefix("HTTP/1.1 100 Continue\r\n\r\n").unwrap_or(reply);
        let (head, body) = reply.split_once("\r\n\r\n").unwrap_or_else(|| panic!("no reply head: {reply:?}"));
        let length = head.lines().find_map(|l| l.strip_prefix("Content-Length: ")).expect("a Content-Length");
        assert_eq!(length.parse::<usize>().unwrap(), body.len(), "{reply:?}");
        assert!(head.starts_with("HTTP/1.1 "), "{reply:?}");
        status(reply)
    }

    const FUZZ_REQUESTS: [&[u8]; 3] = [
        b"POST /api/analyze?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 12\r\n\r\n{\"a\": [1,2]}",
        b"GET /api/info HTTP/1.1\r\nHost: localhost\r\nAccept: */*\r\n\r\n",
        b"POST / HTTP/1.0\r\nExpect: 100-continue\r\nContent-Length: 3\r\n\r\nabc",
    ];

    #[test]
    fn requests_arriving_in_pieces_are_read_whole() {
        let port = echo_server_with(PATIENT);
        let mut rng = Rng::new(1);
        for raw in FUZZ_REQUESTS {
            let whole = exchange_in_pieces(port, raw, raw.len() as u64, &mut rng);
            assert_eq!(checked_status(&whole), 200, "{whole}");
            for piece in [1, 3, 16] {
                for _ in 0..4 {
                    assert_eq!(exchange_in_pieces(port, raw, piece, &mut rng), whole);
                }
            }
        }
    }

    #[test]
    fn mangled_requests_get_a_well_formed_reply() {
        let port = echo_server_with(PATIENT);
        let mut rng = Rng::new(2);
        let mut statuses = std::collections::BTreeSet::new();
        for _ in 0..600 {
            let mut raw = FUZZ_REQUESTS[rng.below(FUZZ_REQUESTS.len() as u64) as usize].to_vec();
            for _ in 0..1 + rng.below(3) {
                let at = rng.below(raw.len() as u64 + 1) as usize;
                let n = 1 + rng.below(8) as usize;
                match rng.below(5) {
                    0 if at < raw.len() => raw[at] = rng.below(256) as u8,
                    1 => {
                        let bytes: Vec<u8> = (0..n).map(|_| b" :\r\n0123456789x-"[rng.below(16) as usize]).collect();
                        drop(raw.splice(at..at, bytes));
                    }
                    2 => drop(raw.drain(at..(at + n).min(raw.len()))),
                    3 => raw.truncate(at),
                    _ => {
                        let copy = raw[at..(at + n).min(raw.len())].to_vec();
                        drop(raw.splice(at..at, copy));
                    }
                }
            }
            // Mostly in one piece: pieces with pauses take time.
            let piece = if rng.below(8) == 0 { 8 } else { raw.len() as u64 + 1 };
            let reply = exchange_in_pieces(port, &raw, piece, &mut rng);
            let status = checked_status(&reply);
            assert!(
                [200, 400, 413, 417, 431, 501, 505].contains(&status),
                "{status} for {:?}",
                String::from_utf8_lossy(&raw)
            );
            statuses.insert(status);
        }
        assert!(statuses.len() >= 5, "{statuses:?}");
        // The server is still whole: every connection slot was given back.
        for _ in 0..2 * LIMITS.connections {
            assert_eq!(status(&exchange(port, b"GET / HTTP/1.1\r\n\r\n")), 200);
        }
    }

    #[test]
    fn slow_clients_time_out_and_do_not_block_others() {
        let port = echo_server();
        let connect = || TcpStream::connect(("127.0.0.1", port)).unwrap();
        // Two connections stall: one in its head, one in its body.
        let mut stalled = [connect(), connect()];
        stalled[0].write_all(b"GET / HTTP/1.1\r\nHost: h").unwrap();
        stalled[1].write_all(b"POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc").unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(status(&exchange(port, b"GET /still-served HTTP/1.1\r\n\r\n")), 200);
        let started = Instant::now();
        for mut stream in stalled {
            let mut reply = String::new();
            stream.read_to_string(&mut reply).unwrap();
            assert_eq!(status(&reply), 408, "{reply}");
        }
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn a_client_cannot_hold_a_connection_after_its_reply() {
        let port = echo_server();
        // Every connection slot is taken by a client that, after its reply, sends a byte
        // every 100 ms for 8 s. The server takes them in for `LINGER` in all, then closes.
        for _ in 0..LIMITS.connections {
            std::thread::spawn(move || {
                let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
                stream.write_all(b"GET / HTTP/1.1\r\n\r\n").unwrap();
                let _ = stream.read_to_end(&mut Vec::new());
                let until = Instant::now() + Duration::from_secs(8);
                while Instant::now() < until && stream.write_all(b"x").is_ok() {
                    std::thread::sleep(Duration::from_millis(100));
                }
            });
        }
        std::thread::sleep(Duration::from_millis(300));
        let started = Instant::now();
        assert_eq!(status(&exchange(port, b"GET /next HTTP/1.1\r\n\r\n")), 200);
        assert!(started.elapsed() < Duration::from_secs(5), "served after {:?}", started.elapsed());
    }

    #[test]
    fn connections_beyond_the_limit_wait() {
        let port = echo_server();
        let idle: Vec<_> = (0..LIMITS.connections).map(|_| TcpStream::connect(("127.0.0.1", port)).unwrap()).collect();
        let waiting = std::thread::spawn(move || exchange(port, b"GET /waited HTTP/1.1\r\n\r\n"));
        std::thread::sleep(Duration::from_millis(200));
        assert!(!waiting.is_finished(), "a connection beyond the limit was served");
        drop(idle);
        assert!(waiting.join().unwrap().ends_with("GET /waited "));
    }
}
