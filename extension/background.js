const HOST = "com.distill.host";
const READER = chrome.runtime.getURL("reader.html");
const PDF_URLS = ["*://*/*.pdf", "*://*/*.pdf?*", "*://*/*.PDF", "file:///*.pdf", "file:///*.PDF"];

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({ id: "distill", title: "Distill", contexts: ["selection"] });
    chrome.contextMenus.create({
      id: "distill-reader-page",
      title: "Open in Distill reader",
      contexts: ["page", "frame"],
      documentUrlPatterns: PDF_URLS,
    });
    chrome.contextMenus.create({
      id: "distill-reader-link",
      title: "Open PDF in Distill reader",
      contexts: ["link"],
      targetUrlPatterns: PDF_URLS,
    });
  });
});

const readerUrl = (pdf) => `${READER}?file=${encodeURIComponent(pdf)}`;

// The toolbar button opens the current PDF in the reader.
chrome.action.onClicked.addListener(async (tab) => {
  if (tab.id && (await isPdf(tab.url))) chrome.tabs.update(tab.id, { url: readerUrl(tab.url) });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab?.id) return;
  if (info.menuItemId === "distill-reader-link") {
    chrome.tabs.create({ url: readerUrl(info.linkUrl), index: tab.index + 1 });
    return;
  }
  if (info.menuItemId === "distill-reader-page") {
    chrome.tabs.update(tab.id, { url: readerUrl(info.frameUrl || tab.url) });
    return;
  }
  if (info.menuItemId !== "distill") return;
  if (tab.url?.startsWith(READER)) return; // the reader page handles its own clicks

  const frameId = info.frameId ?? 0;
  const send = (msg) => chrome.tabs.sendMessage(tab.id, msg, { frameId });

  let captured;
  try {
    captured = await send({ type: "distill-capture" });
  } catch {
    // Tabs opened before the extension was installed or reloaded have no
    // content script yet, so inject it now and try again.
    try {
      const target = { tabId: tab.id, frameIds: [frameId] };
      await chrome.scripting.insertCSS({ target, files: ["content.css"] });
      await chrome.scripting.executeScript({ target, files: ["content.js"] });
      captured = await send({ type: "distill-capture" });
    } catch (e) {
      // Chrome's PDF viewer can't be scripted, so reopen the PDF in the
      // Distill reader and distill the same passage there.
      const pdf = info.frameUrl || info.pageUrl || tab.url;
      if (info.selectionText && (await isPdf(pdf))) {
        await chrome.storage.session.set({ [`pending:${tab.id}`]: info.selectionText });
        chrome.tabs.update(tab.id, { url: readerUrl(pdf) });
      } else {
        console.warn("Distill: can't run on this page:", e.message);
      }
      return;
    }
  }
  if (!captured?.text) return;

  let reply;
  try {
    reply = await chrome.runtime.sendNativeMessage(HOST, { text: captured.text });
  } catch (e) {
    reply = { ok: false, error: `can't reach the distill helper (${e.message}). Run install.sh.` };
  }
  send({ type: "distill-apply", id: captured.id, ...reply }).catch(() => {});
});

async function isPdf(url) {
  if (!url || !/^(https?|file):/.test(url)) return false;
  if (/\.pdf($|[?#])/i.test(url)) return true;
  if (url.startsWith("file:")) return false;
  try {
    const res = await fetch(url, { method: "HEAD" });
    return /application\/pdf/i.test(res.headers.get("content-type") || "");
  } catch {
    return false;
  }
}
