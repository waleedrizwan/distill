// Distill reader: shows a PDF as plain, selectable text so passages can be
// distilled in place (Chrome's own PDF viewer doesn't let extensions edit it).

import * as pdfjs from "./vendor/pdf.min.mjs";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("./vendor/pdf.worker.min.mjs", import.meta.url).href;

const HOST = "com.distill.host";
const params = new URLSearchParams(location.search);
const file = params.get("file");
const doc = document.getElementById("doc");
const status = document.getElementById("status");
const tabPromise = chrome.tabs.getCurrent();

document.getElementById("undo-key").textContent = /Mac/.test(navigator.platform) ? "⌘Z" : "Ctrl+Z";

// Right-click > Distill inside the reader. The background script leaves these
// clicks to us because this page, not a content script, owns the text.
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "distill" || tab?.id !== (await tabPromise)?.id) return;
  distillSelection();
});

main().catch((e) => {
  console.error(e);
  status.className = "error";
  status.textContent = `Couldn't open this PDF: ${e.message}`;
});

async function main() {
  if (!file) throw new Error("no file given");
  const url = new URL(file);
  document.getElementById("source").textContent = url.href;
  document.getElementById("source").href = url.href;
  document.getElementById("original").href = url.href;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed (HTTP ${res.status})`);
  const data = new Uint8Array(await res.arrayBuffer());
  const pdf = await pdfjs.getDocument({ data }).promise;

  const pages = [];
  for (let n = 1; n <= pdf.numPages; n++) {
    status.lastChild.textContent = ` Extracting text… page ${n} of ${pdf.numPages}`;
    const page = await pdf.getPage(n);
    pages.push(linesOf(await page.getTextContent()));
  }

  render(pages);

  const meta = await pdf.getMetadata().catch(() => null);
  const title =
    meta?.info?.Title?.trim() ||
    doc.querySelector("h1")?.textContent ||
    decodeURIComponent(url.pathname.split("/").pop() || url.host);
  document.title = `${title} · Distill reader`;
  document.getElementById("title").textContent = title;

  if (!doc.textContent.trim()) {
    throw new Error("no selectable text in it (it may be a scanned image)");
  }
  await distillPending();
}

// ---- text extraction -------------------------------------------------------

/** Groups a page's text items into lines, in the PDF's own reading order. */
function linesOf(content) {
  const lines = [];
  let cur = null;
  for (const it of content.items) {
    if (!("str" in it)) continue;
    const [, , , , x, y] = it.transform;
    const size = Math.hypot(it.transform[2], it.transform[3]) || it.height || 10;
    if (!cur || Math.abs(y - cur.y) > size * 0.5) {
      if (cur && cur.text.trim()) lines.push(cur);
      cur = { x, y, size, text: "" };
    } else if (it.str && cur.text && !/\s$/.test(cur.text) && x - cur.end > size * 0.15) {
      cur.text += " ";
    }
    cur.text += it.str;
    cur.size = Math.max(cur.size, it.str.trim() ? size : 0);
    cur.end = x + (it.width || 0);
    if (it.hasEOL) {
      if (cur.text.trim()) lines.push(cur);
      cur = null;
    }
  }
  if (cur && cur.text.trim()) lines.push(cur);
  return lines.map((l) => ({ ...l, text: l.text.replace(/\s+/g, " ").trim() }));
}

function render(pages) {
  const all = pages.flat();
  const body = mode(all.filter((l) => l.text.length > 30).map((l) => Math.round(l.size)));
  const gaps = [];
  for (const lines of pages) {
    for (let i = 1; i < lines.length; i++) {
      const d = lines[i - 1].y - lines[i].y;
      if (d > 0 && d < body * 3) gaps.push(d);
    }
  }
  const lineGap = median(gaps) || body * 1.2;

  doc.replaceChildren();
  let para = null;
  let prev = null;
  const flush = () => {
    if (para?.text) {
      const el = document.createElement(para.heading ? (doc.childElementCount ? "h2" : "h1") : "p");
      el.textContent = para.text;
      doc.append(el);
    }
    para = null;
  };

  pages.forEach((lines, pi) => {
    if (pi > 0) {
      flush();
      const hr = document.createElement("hr");
      hr.className = "page-break";
      hr.dataset.page = `page ${pi + 1}`;
      doc.append(hr);
      prev = null;
    }
    for (const line of lines) {
      if (/^\d{1,4}$/.test(line.text)) continue; // page numbers
      const heading =
        (line.size > body * 1.15 && line.text.split(" ").length <= 14 && /[A-Za-z]{3}/.test(line.text)) ||
        /^(\d+(\.\d+)*\.?|[IVX]+\.)\s+[A-Z][\w ,:&()'-]{1,60}$/.test(line.text); // "2. Transactions"
      const gap = prev ? prev.y - line.y : 0;
      const newPara =
        !para ||
        heading !== para.heading ||
        gap < 0 || // jumped up: next column
        gap > lineGap * 1.45 ||
        (para.heading && gap > lineGap * 1.1);
      if (newPara) {
        flush();
        para = { heading, text: line.text };
      } else if (/[a-z]-$/.test(para.text) && /^[a-z]/.test(line.text)) {
        para.text = para.text.slice(0, -1) + line.text; // re-join hyphenated words
      } else {
        para.text += " " + line.text;
      }
      prev = line;
    }
  });
  flush();
}

function mode(xs) {
  const counts = new Map();
  let best = 10;
  for (const x of xs) {
    counts.set(x, (counts.get(x) || 0) + 1);
    if (counts.get(x) > (counts.get(best) || 0)) best = x;
  }
  return best;
}

function median(xs) {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

// ---- distilling ------------------------------------------------------------

async function distillSelection() {
  const captured = window.__distill.capture();
  if (!captured) return;
  let reply;
  try {
    reply = await chrome.runtime.sendNativeMessage(HOST, { text: captured.text });
  } catch (e) {
    reply = { ok: false, error: `can't reach the distill helper (${e.message}). Run install.sh.` };
  }
  window.__distill.apply({ id: captured.id, ...reply });
  const outs = doc.querySelectorAll(".distill-out");
  outs[outs.length - 1]?.scrollIntoView({ block: "center", behavior: "smooth" });
}

/** If we were opened from Chrome's PDF viewer, distill the text picked there. */
async function distillPending() {
  const tab = await tabPromise;
  const key = `pending:${tab?.id}`;
  const { [key]: text } = await chrome.storage.session.get(key);
  if (!text) return;
  await chrome.storage.session.remove(key);
  const range = findText(text);
  if (!range) {
    window.__distill.toast?.({ text: "Couldn't find that selection in the extracted text. Highlight it here and try again.", kind: "warn" });
    return;
  }
  const sel = getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
  range.startContainer.parentElement.scrollIntoView({ block: "center" });
  await distillSelection();
}

/** Finds text in the reader, ignoring spacing, punctuation, hyphens and ligatures. */
function findText(needle) {
  const key = (s) => s.normalize("NFKD").toLowerCase().replace(/[^a-z0-9]/g, "");
  const map = []; // index in the squashed haystack -> [text node, offset]
  let hay = "";
  const walker = document.createTreeWalker(doc, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    for (let i = 0; i < n.data.length; i++) {
      const k = key(n.data[i]);
      for (const ch of k) {
        hay += ch;
        map.push([n, i]);
      }
    }
  }
  const k = key(needle);
  const at = k ? hay.indexOf(k) : -1;
  if (at < 0) return null;
  const [sn, so] = map[at];
  const [en, eo] = map[at + k.length - 1];
  // The match skips punctuation, so take back any the selection had at its edges.
  let start = so;
  let end = eo + 1;
  const lead = needle.trimStart().match(/^[^\p{L}\p{N}]*/u)[0].length;
  const tail = needle.trimEnd().match(/[^\p{L}\p{N}]*$/u)[0].length;
  for (let i = 0; i < lead && start > 0 && /[^\p{L}\p{N}\s]/u.test(sn.data[start - 1]); i++) start--;
  for (let i = 0; i < tail && end < en.data.length && /[^\p{L}\p{N}\s]/u.test(en.data[end]); i++) end++;
  const r = document.createRange();
  r.setStart(sn, start);
  r.setEnd(en, end);
  return r;
}
