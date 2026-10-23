#!/usr/bin/env bash
# Checks that distill's output is reproducible.
#
#   eval/determinism.sh            compare against eval/golden.sha256
#   eval/determinism.sh --update   rewrite eval/golden.sha256 after an intended change
#   eval/determinism.sh --threads-only   skip the golden comparison
#
# Each committed sample is summarized in several modes and hashed. The run is
# repeated with 1 and 4 threads, which must agree with each other. The golden
# hashes were recorded on macOS arm64; another CPU type can differ in the last
# float digits of the embeddings, so CI compares goldens on the same platform
# and only checks thread-independence elsewhere.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=${DISTILL_BIN:-target/release/distill}
GOLDEN=eval/golden.sha256
MODES=("" "-n 3" "--words 150" "--centrality centroid" "--no-repair --lambda 0.5" "--compress --words 150" "--simplify --compress --words 150" "--split --compress --simplify --words 150")
mode=${1:-check}

[ -x "$BIN" ] || { echo "$BIN not found; run: cargo build --release" >&2; exit 1; }
sha() { if command -v shasum >/dev/null 2>&1; then shasum -a 256 | cut -d' ' -f1; else sha256sum | cut -d' ' -f1; fi; }

hashes() { # threads
  for f in samples/*.txt samples/*.pdf; do
    for m in "${MODES[@]}"; do
      # shellcheck disable=SC2086
      h=$(env -u DISTILL_CACHE RAYON_NUM_THREADS="$1" "$BIN" $m "$f" | sha)
      printf '%s  %s %s\n' "$h" "$f" "${m:-(defaults)}"
    done
  done
  # The text a PDF is read as, before any summarizing.
  for f in samples/*.pdf; do
    h=$("$BIN" --extract-text "$f" | sha)
    printf '%s  %s %s\n' "$h" "$f" "--extract-text"
  done
}

one=$(hashes 1)
four=$(hashes 4)
if [ "$one" != "$four" ]; then
  echo "output differs between 1 and 4 threads:" >&2
  diff <(echo "$one") <(echo "$four") >&2 || true
  exit 1
fi
echo "1 and 4 threads agree on $(echo "$one" | wc -l | tr -d ' ') outputs"

case "$mode" in
  --update)
    echo "$one" > "$GOLDEN"
    echo "wrote $GOLDEN"
    ;;
  --threads-only) ;;
  check)
    if ! diff <(cat "$GOLDEN") <(echo "$one") >/dev/null; then
      echo "output changed from $GOLDEN (if intended, run: eval/determinism.sh --update):" >&2
      diff "$GOLDEN" <(echo "$one") >&2 || true
      exit 1
    fi
    echo "all outputs match $GOLDEN"
    ;;
  *) echo "unknown option $mode" >&2; exit 2 ;;
esac
