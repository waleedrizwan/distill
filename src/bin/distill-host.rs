//! Chrome native-messaging host for the distill browser extension.
//!
//! Chrome starts this binary, writes one length-prefixed JSON message to stdin
//! ({"text": "...", "ratio": 0.3} or {"text": "...", "sentences": 4}), and reads
//! one length-prefixed JSON reply from stdout ({"ok": true, "summary": "..."}).
//! The summarizing itself is done by the `distill` binary sitting next to this one.

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

fn summarize(req: &Value) -> Result<String, String> {
    let text = req.get("text").and_then(Value::as_str).ok_or("no text in request")?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let distill = exe.with_file_name(if cfg!(windows) { "distill.exe" } else { "distill" });

    let mut cmd = Command::new(&distill);
    cmd.arg("--no-headings");
    if let Some(n) = req.get("sentences").and_then(Value::as_u64) {
        cmd.args(["--sentences", &n.to_string()]);
    } else if let Some(r) = req.get("ratio").and_then(Value::as_f64) {
        cmd.args(["--ratio", &r.to_string()]);
    }
    let mut child = cmd
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("starting {}: {e}", distill.display()))?;
    child.stdin.take().unwrap().write_all(text.as_bytes()).map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_string())
}

fn main() {
    let reply = match read_message() {
        Ok(req) => match summarize(&req) {
            Ok(summary) => json!({ "ok": true, "summary": summary }),
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
