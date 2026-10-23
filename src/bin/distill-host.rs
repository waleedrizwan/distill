//! Chrome native-messaging host for the distill browser extension.
//!
//! Chrome starts this binary, writes one length-prefixed JSON message to stdin,
//! and reads one length-prefixed JSON reply from stdout. A request holds either
//! `text` (the highlighted passage) or `pdf` (a whole PDF, base64), and
//! optionally `sentences` or `ratio` (length), `by_section` (sentences per
//! section), `plain` and `glossary` (booleans, the CLI flags of those names).
//! The reply is {"ok": true, "summary": "...", "glossary": [{"term", "definition"}]}
//! or {"ok": false, "error": "..."}. The summarizing itself is done by the
//! `distill` binary sitting next to this one.

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::process::{Command, Stdio};

fn read_message() -> std::io::Result<Value> {
    let mut len = [0u8; 4];
    std::io::stdin().read_exact(&mut len)?;
    let mut buf = vec![0u8; u32::from_ne_bytes(len) as usize];
    std::io::stdin().read_exact(&mut buf)?;
    serde_json::from_slice(&buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn write_message(v: &Value) -> std::io::Result<()> {
    let body = serde_json::to_vec(v)?;
    let mut out = std::io::stdout().lock();
    out.write_all(&(body.len() as u32).to_ne_bytes())?;
    out.write_all(&body)?;
    out.flush()
}

/// Standard base64 (padding optional, whitespace ignored), for `pdf`.
fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let value = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes().filter(|c| !c.is_ascii_whitespace() && *c != b'=') {
        acc = (acc << 6) | value(c).ok_or("the PDF is not valid base64")?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

/// The command-line flags a request asks for.
fn flags(req: &Value) -> Vec<String> {
    let mut args = vec!["--no-headings".to_string()];
    let on = |k: &str| req.get(k).and_then(Value::as_bool).unwrap_or(false);
    if let Some(n) = req.get("by_section").and_then(Value::as_u64) {
        // Headings are what make a section-by-section summary readable.
        args = vec![format!("--by-section={}", n.max(1))];
    } else if let Some(n) = req.get("sentences").and_then(Value::as_u64) {
        args.extend(["--sentences".to_string(), n.to_string()]);
    } else if let Some(r) = req.get("ratio").and_then(Value::as_f64) {
        args.extend(["--ratio".to_string(), r.to_string()]);
    }
    if on("plain") {
        args.push("--plain".into());
    }
    if on("glossary") {
        args.push("--glossary".into());
    }
    args
}

/// Splits distill's text output into the summary and its glossary entries.
fn split_glossary(out: &str) -> (String, Vec<Value>) {
    match out.split_once("\n\nGlossary\n") {
        Some((summary, gloss)) => {
            let entries = gloss
                .lines()
                .filter_map(|l| l.trim().split_once(": "))
                .map(|(term, definition)| json!({ "term": term, "definition": definition }))
                .collect();
            (summary.trim_end().to_string(), entries)
        }
        None => (out.trim_end().to_string(), Vec::new()),
    }
}

fn summarize(req: &Value) -> Result<(String, Vec<Value>), String> {
    let input: Vec<u8> = match (req.get("text").and_then(Value::as_str), req.get("pdf").and_then(Value::as_str)) {
        (Some(text), _) => text.as_bytes().to_vec(),
        (None, Some(pdf)) => base64_decode(pdf)?,
        (None, None) => return Err("no text in request".into()),
    };
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let distill = exe.with_file_name(if cfg!(windows) { "distill.exe" } else { "distill" });

    let mut child = Command::new(&distill)
        .args(flags(req))
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("starting {}: {e}", distill.display()))?;
    child.stdin.take().unwrap().write_all(&input).map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(split_glossary(&String::from_utf8_lossy(&out.stdout)))
}

fn main() {
    let reply = match read_message() {
        Ok(req) => match summarize(&req) {
            Ok((summary, glossary)) => json!({ "ok": true, "summary": summary, "glossary": glossary }),
            Err(error) => json!({ "ok": false, "error": error }),
        },
        Err(e) => json!({ "ok": false, "error": format!("bad message: {e}") }),
    };
    log(&reply);
    let _ = write_message(&reply);
}

/// Appends one line per request to ~/Library/Logs/distill-host.log (or
/// ~/.distill/host.log elsewhere), since Chrome hides the host's stderr.
fn log(reply: &Value) {
    let Some(home) = std::env::var_os("HOME") else { return };
    let home = std::path::PathBuf::from(home);
    let path = if cfg!(target_os = "macos") {
        home.join("Library/Logs/distill-host.log")
    } else {
        home.join(".distill/host.log")
    };
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let status = match reply.get("error").and_then(Value::as_str) {
        Some(e) => format!("error: {e}"),
        None => "ok".to_string(),
    };
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{secs} {status}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips() {
        assert_eq!(base64_decode("JVBERi0xLjQ=").unwrap(), b"%PDF-1.4");
        assert_eq!(base64_decode("aGVsbG8gd29y\nbGQ").unwrap(), b"hello world");
        assert!(base64_decode("not*base64").is_err());
    }

    #[test]
    fn request_flags() {
        assert_eq!(flags(&json!({"text": "x", "ratio": 0.3})), ["--no-headings", "--ratio", "0.3"]);
        assert_eq!(flags(&json!({"text": "x", "plain": true, "glossary": true})), ["--no-headings", "--plain", "--glossary"]);
        assert_eq!(flags(&json!({"pdf": "x", "by_section": 2})), ["--by-section=2"]);
    }

    #[test]
    fn glossary_split_off() {
        let (s, g) = split_glossary("First line.\nSecond line.\n\nGlossary\n  LLR: log-likelihood ratio\n  SVM: support vector machine\n");
        assert_eq!(s, "First line.\nSecond line.");
        assert_eq!(g, vec![json!({"term": "LLR", "definition": "log-likelihood ratio"}), json!({"term": "SVM", "definition": "support vector machine"})]);
        assert_eq!(split_glossary("Only.\n"), ("Only.".to_string(), vec![]));
    }
}
