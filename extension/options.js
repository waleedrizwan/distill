// Two switches, kept in chrome.storage.sync and read by background.js and reader.js.
const DEFAULTS = { plain: false, glossary: true };

chrome.storage.sync.get(DEFAULTS, (settings) => {
  for (const [key, value] of Object.entries(settings)) {
    const box = document.getElementById(key);
    box.checked = value;
    box.addEventListener("change", () => chrome.storage.sync.set({ [key]: box.checked }));
  }
});
