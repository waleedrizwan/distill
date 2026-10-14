#!/usr/bin/env bash
# Downloads a sentence-embedding model. Revisions are pinned and every file is
# checksum-verified, so every machine ranks sentences with exactly the same
# weights.
#
#   ./fetch-model.sh                 all-MiniLM-L6-v2 into ./model (the default)
#   ./fetch-model.sh NAME [DIR]      another model, into ./model-NAME by default
#   ./fetch-model.sh DIR             the default model into DIR
#
# NAME is one of: minilm, bge, gte, e5. Use another model with --model-dir.
#
# Alongside the weights, distill.json records how the model must be used:
# pooling (mean over tokens, or the CLS token), the token limit it was trained
# with, and any text prefix it expects.
set -euo pipefail

NAME=minilm
case "${1:-}" in
  minilm|bge|gte|e5) NAME=$1; shift ;;
esac

case "$NAME" in
  minilm) # all-MiniLM-L6-v2, ~90 MB, Apache-2.0, 6 layers
    REPO=sentence-transformers/all-MiniLM-L6-v2
    REV=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
    SHA_CONFIG=953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41
    SHA_TOKENIZER=be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037
    SHA_WEIGHTS=53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
    SETTINGS='{"pooling": "mean", "max_tokens": 256, "prefix": ""}'
    DEFAULT_DIR=model ;;
  bge) # BAAI/bge-small-en-v1.5, ~130 MB, MIT, 12 layers
    REPO=BAAI/bge-small-en-v1.5
    REV=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
    SHA_CONFIG=094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750
    SHA_TOKENIZER=d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66
    SHA_WEIGHTS=3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad
    SETTINGS='{"pooling": "cls", "max_tokens": 512, "prefix": ""}'
    DEFAULT_DIR=model-bge ;;
  gte) # thenlper/gte-small, ~65 MB (float16), MIT, 12 layers
    REPO=thenlper/gte-small
    REV=17e1f347d17fe144873b1201da91788898c639cd
    SHA_CONFIG=00273cdfb3e05c7304d180270bdef8adae45023e7bdfc396338bdd815943dc59
    SHA_TOKENIZER=da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0
    SHA_WEIGHTS=9a1eb90bbac323ea08aa5629b624fe6ae75db121b904799c2266a1e2c2de22d2
    SETTINGS='{"pooling": "mean", "max_tokens": 512, "prefix": ""}'
    DEFAULT_DIR=model-gte ;;
  e5) # intfloat/e5-small-v2, ~130 MB, MIT, 12 layers; "query: " for symmetric similarity
    REPO=intfloat/e5-small-v2
    REV=ffb93f3bd4047442299a41ebb6fa998a38507c52
    SHA_CONFIG=5dfb0363cd0243be179c03bcaafd1542d0fbb95e8cbcf575fff3e229342adc2f
    SHA_TOKENIZER=d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66
    SHA_WEIGHTS=45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1
    SETTINGS='{"pooling": "mean", "max_tokens": 512, "prefix": "query: "}'
    DEFAULT_DIR=model-e5 ;;
esac

BASE="https://huggingface.co/$REPO/resolve/$REV"
DIR="${1:-$DEFAULT_DIR}"
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

fetch config.json       "$SHA_CONFIG"
fetch tokenizer.json    "$SHA_TOKENIZER"
fetch model.safetensors "$SHA_WEIGHTS"
echo "$SETTINGS" > "$DIR/distill.json"
echo "$NAME model ready in $DIR/"
