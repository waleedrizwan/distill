#!/usr/bin/env bash
# Builds distill and registers it with Chrome so the extension can call it.
# Then load ./extension in chrome://extensions (Developer mode > Load unpacked).
set -euo pipefail
cd "$(dirname "$0")"

./fetch-model.sh
# On macOS the build uses Apple's Accelerate framework automatically.
cargo build --release

# Fixed by the "key" in extension/manifest.json.
EXT_ID=hdjlniolemieldophfomihofgmbiplfo
HOST_PATH="$PWD/target/release/distill-host"

if [ "$(uname -s)" = Darwin ]; then
  DIR="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts"
else
  DIR="$HOME/.config/google-chrome/NativeMessagingHosts"
fi
mkdir -p "$DIR"
cat > "$DIR/com.distill.host.json" <<JSON
{
  "name": "com.distill.host",
  "description": "distill extractive summarizer",
  "path": "$HOST_PATH",
  "type": "stdio",
  "allowed_origins": ["chrome-extension://$EXT_ID/"]
}
JSON
echo "registered $DIR/com.distill.host.json"
echo "now load $PWD/extension in chrome://extensions (Developer mode > Load unpacked)"
