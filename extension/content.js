// Captures the highlighted text, then swaps it for distill's summary.
// Click a distilled passage to flip between the summary and the original;
// Ctrl+Z / Cmd+Z reverts the most recent distill.

if (!window.__distillLoaded) {
  window.__distillLoaded = true;
  (() => {
    const pending = new Map();
    const history = []; // distilled page passages, newest last, for Ctrl+Z
    let nextId = 1;

    chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
      if (msg.type === "distill-capture") {
        reply(capture());
      } else if (msg.type === "distill-apply") {
        apply(msg);
      }
    });

    // The Distill reader page drives these directly.
    window.__distill = { capture, apply, toast };

    document.addEventListener(
      "keydown",
      (e) => {
        if (e.key.toLowerCase() !== "z" || e.shiftKey || e.altKey || !(e.ctrlKey || e.metaKey)) return;
        // Inside text boxes and editors the browser's own undo already covers
        // distill, because the text went in through insertText.
        if (isEditable(document.activeElement)) return;
        if (undoLast()) {
          e.preventDefault();
          e.stopPropagation();
        }
      },
      true
    );

    function capture() {
      const el = document.activeElement;
      let entry;
      if (el && isTextField(el) && el.selectionStart !== el.selectionEnd) {
        entry = { field: el, start: el.selectionStart, end: el.selectionEnd };
        entry.text = el.value.slice(entry.start, entry.end);
      } else {
        const sel = window.getSelection();
        if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
        entry = { range: sel.getRangeAt(0).cloneRange(), text: sel.toString() };
        markPending(entry.range);
      }
      if (!entry.text.trim()) return null;
      const id = nextId++;
      pending.set(id, entry);
      entry.toast = toast({ text: "Distilling…", spinner: true, sticky: true });
      return { id, text: entry.text };
    }

    function apply({ id, ok, summary, error }) {
      const entry = pending.get(id);
      if (!entry) return;
      pending.delete(id);
      entry.toast.dismiss();
      clearPending();

      if (!ok) return toast({ text: `Distill failed: ${error}`, kind: "error" });
      if (!summary.trim()) return toast({ text: "Nothing worth keeping in that selection.", kind: "warn" });

      // distill puts sentences that were adjacent in the source on one line;
      // a new line means text was skipped in between.
      const lines = summary.split("\n").filter((l) => l.trim());
      if (squash(lines.join(" ")) === squash(entry.text)) {
        return toast({ text: "Already as short as it gets. Try selecting more text.", kind: "warn" });
      }
      const before = words(entry.text);
      const after = words(lines.join(" "));
      const stats = `${before} → ${after} words (−${Math.round(100 - (100 * after) / before)}%)`;
      const undoKey = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘Z" : "Ctrl+Z";

      if (entry.field) {
        const joined = lines.join(entry.field.tagName === "TEXTAREA" ? "\n\n" : " … ");
        entry.field.focus();
        entry.field.setSelectionRange(entry.start, entry.end);
        // insertText keeps the field's native undo history, so Ctrl+Z works there.
        if (!document.execCommand("insertText", false, joined)) {
          entry.field.setRangeText(joined, entry.start, entry.end, "end");
          entry.field.dispatchEvent(new Event("input", { bubbles: true }));
        }
        flash(entry.field);
        return toast({ text: `Distilled: ${stats}`, hint: `${undoKey} to undo`, kind: "done" });
      }

      const range = entry.range;
      const editor = closestEditable(range.commonAncestorContainer);
      if (editor) {
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand("insertText", false, lines.join(/\n/.test(entry.text) ? "\n\n" : " … "));
        return toast({ text: `Distilled: ${stats}`, hint: `${undoKey} to undo`, kind: "done" });
      }

      // Keep a copy of the untouched passage so undo puts back exactly what was there.
      const host = asElement(range.commonAncestorContainer);
      const startPath = pathTo(host, range.startContainer);
      const endPath = pathTo(host, range.endContainer);
      const startOffset = range.startOffset;
      const endOffset = range.endOffset;
      const originalNodes = [...host.childNodes].map((n) => n.cloneNode(true));
      const startBlock = blockOf(range.startContainer, host);
      const endBlock = blockOf(range.endContainer, host);

      range.deleteContents();
      const multiBlock = /\n/.test(entry.text);
      const out = document.createElement("span");
      out.className = "distill-out distill-enter";
      out.title = "Distilled. Click to show the original.";
      lines.forEach((line, i) => {
        if (i) {
          if (multiBlock) out.append(document.createElement("br"), document.createElement("br"));
          else {
            const gap = document.createElement("span");
            gap.className = "distill-gap";
            gap.textContent = " … ";
            out.append(gap);
          }
        }
        out.append(line);
      });
      // Put the summary in the paragraph the selection started in, rather than
      // between paragraphs, and drop a trailing paragraph the selection emptied.
      if (startBlock !== host && startBlock.isConnected) startBlock.append(out);
      else range.insertNode(out);
      if (endBlock !== host && endBlock !== startBlock && isBlank(endBlock)) endBlock.remove();
      window.getSelection()?.removeAllRanges();
      out.addEventListener("animationend", () => out.classList.remove("distill-enter"), { once: true });

      const record = {
        host,
        originalNodes,
        distilledNodes: [...host.childNodes],
        showingOriginal: false,
        originalRange() {
          const r = document.createRange();
          r.setStart(follow(host, startPath), startOffset);
          r.setEnd(follow(host, endPath), endOffset);
          return r;
        },
      };
      history.push(record);
      out.addEventListener("click", (e) => {
        if (window.getSelection()?.toString()) return; // user is selecting, not toggling
        e.preventDefault();
        e.stopPropagation();
        toggle(record);
      });

      toast({
        text: `Distilled: ${stats}`,
        hint: `${undoKey} to undo · click text to compare`,
        kind: "done",
        action: { label: "Undo", run: () => revert(record) },
      });
    }

    // Clicking a passage that is showing its original flips it back to the summary.
    document.addEventListener(
      "click",
      (e) => {
        if (window.getSelection()?.toString()) return;
        const pos = caretAt(e.clientX, e.clientY);
        if (!pos) return;
        const r = history.find(
          (r) => r.showingOriginal && r.host.isConnected && r.originalRange().isPointInRange(pos.offsetNode, pos.offset)
        );
        if (!r) return;
        e.preventDefault();
        e.stopPropagation();
        toggle(r);
      },
      true
    );

    function toggle(r) {
      r.showingOriginal = !r.showingOriginal;
      r.host.replaceChildren(...(r.showingOriginal ? r.originalNodes : r.distilledNodes));
      paintOriginals();
    }

    function revert(r) {
      const i = history.indexOf(r);
      if (i < 0) return false;
      history.splice(i, 1);
      if (!r.host.isConnected) return false;
      if (!r.showingOriginal) r.host.replaceChildren(...r.originalNodes);
      r.showingOriginal = false;
      paintOriginals();
      toast({ text: "Restored the original text.", kind: "undo" });
      return true;
    }

    function undoLast() {
      while (history.length) {
        if (revert(history[history.length - 1])) return true;
      }
      return false;
    }

    // Tint passages that are temporarily showing their original text.
    function paintOriginals() {
      if (!window.CSS?.highlights || typeof Highlight === "undefined") return;
      const ranges = history.filter((r) => r.showingOriginal && r.host.isConnected).map((r) => r.originalRange());
      if (ranges.length) CSS.highlights.set("distill-original", new Highlight(...ranges));
      else CSS.highlights.delete("distill-original");
    }

    function caretAt(x, y) {
      const p = document.caretPositionFromPoint?.(x, y);
      if (p) return p;
      const r = document.caretRangeFromPoint?.(x, y);
      return r && { offsetNode: r.startContainer, offset: r.startOffset };
    }

    function asElement(node) {
      return node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
    }

    function blockOf(node, host) {
      for (let el = asElement(node); el && el !== host; el = el.parentElement) {
        if (!getComputedStyle(el).display.startsWith("inline")) return el;
      }
      return host;
    }

    function isBlank(el) {
      return !el.textContent.trim() && !el.querySelector("img,video,iframe,svg,canvas,input");
    }

    // Child-index path from host down to node, so the same spot can be found
    // again in the restored copy.
    function pathTo(host, node) {
      const path = [];
      for (let n = node; n !== host; n = n.parentNode) path.unshift([...n.parentNode.childNodes].indexOf(n));
      return path;
    }

    function follow(host, path) {
      return path.reduce((n, i) => n.childNodes[i], host);
    }

    // Tint the selection while the summary is being worked out.
    function markPending(range) {
      if (!window.CSS?.highlights || typeof Highlight === "undefined") return;
      CSS.highlights.set("distill-pending", new Highlight(range));
      window.getSelection()?.removeAllRanges();
    }

    function clearPending() {
      if (pending.size === 0) window.CSS?.highlights?.delete("distill-pending");
    }

    function flash(el) {
      el.classList.remove("distill-flash");
      void el.offsetWidth;
      el.classList.add("distill-flash");
      el.addEventListener("animationend", () => el.classList.remove("distill-flash"), { once: true });
    }

    function toast({ text, hint, kind = "info", spinner = false, sticky = false, action }) {
      let stack = document.getElementById("distill-toasts");
      if (!stack) {
        stack = document.createElement("div");
        stack.id = "distill-toasts";
        document.documentElement.append(stack);
      }
      const t = document.createElement("div");
      t.className = `distill-toast distill-${kind}`;
      const icon = document.createElement("span");
      icon.className = spinner ? "distill-spinner" : "distill-icon";
      icon.textContent = spinner ? "" : { done: "✓", error: "!", warn: "i", undo: "↶", info: "i" }[kind];
      const body = document.createElement("div");
      body.className = "distill-body";
      body.append(text);
      if (hint) {
        const h = document.createElement("div");
        h.className = "distill-hint";
        h.textContent = hint;
        body.append(h);
      }
      t.append(icon, body);
      if (action) {
        const b = document.createElement("button");
        b.className = "distill-action";
        b.textContent = action.label;
        b.addEventListener("click", () => {
          action.run();
          dismiss();
        });
        t.append(b);
      }
      stack.append(t);

      let timer;
      function dismiss() {
        clearTimeout(timer);
        t.classList.add("distill-leave");
        setTimeout(() => t.remove(), 200);
      }
      if (!sticky) {
        const ms = kind === "error" ? 8000 : 5000;
        timer = setTimeout(dismiss, ms);
        t.addEventListener("mouseenter", () => clearTimeout(timer));
        t.addEventListener("mouseleave", () => (timer = setTimeout(dismiss, 2000)));
      }
      return { dismiss };
    }

    function isTextField(el) {
      if (el.tagName === "TEXTAREA") return true;
      return el.tagName === "INPUT" && /^(text|search|url|email|)$/i.test(el.type);
    }

    function isEditable(el) {
      return !!el && (isTextField(el) || el.isContentEditable);
    }

    function closestEditable(node) {
      const el = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
      return el?.isContentEditable ? el : null;
    }

    function squash(t) {
      return t.replace(/\s+/g, " ").trim();
    }

    function words(t) {
      return squash(t).split(" ").filter(Boolean).length || 1;
    }
  })();
}
