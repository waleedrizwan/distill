#!/usr/bin/env bash
# Downloads the sentence-embedding model (all-MiniLM-L6-v2, ~90 MB, Apache-2.0)
# into ./model. The revision is pinned and the files are checksum-verified, so
# every machine ranks sentences with exactly the same weights.
set -euo pipefail

REV=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
BASE="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$REV"
DIR="${1:-model}"
mkdir -p "$DIR"

fetch() { # name sha256
  if [ ! -f "$DIR/$1" ]; then
    echo "downloading $1"
    curl -fL --retry 3 -o "$DIR/$1.part" "$BASE/$1"
    mv "$DIR/$1.part" "$DIR/$1"
  fi
  if command -v shasum >/dev/null 2>&1; then
    got=$(shasum -a 256 "$DIR/$1" | cut -d' ' -f1)
  else
    got=$(sha256sum "$DIR/$1" | cut -d' ' -f1)
  fi
  if [ "$got" != "$2" ]; then
    echo "checksum mismatch for $1 (delete $DIR/$1 and re-run)" >&2
    exit 1
  fi
}

fetch config.json       953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41
fetch tokenizer.json    be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037
fetch model.safetensors 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
echo "model ready in $DIR/"
