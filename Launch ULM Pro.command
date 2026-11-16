#!/bin/zsh
cd "$(dirname "$0")"

PORT=8765
URL="http://127.0.0.1:${PORT}"

# Start local HTTP server if not already listening on PORT
if ! lsof -iTCP:${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
  python3 -m http.server ${PORT} --bind 127.0.0.1 >/dev/null 2>&1 &
  SERVER_PID=$!
  sleep 0.4
fi

# Open as standalone app window if Chrome / Edge / Brave is installed
if [ -d "/Applications/Google Chrome.app" ]; then
  open -na "Google Chrome" --args --app="${URL}" --window-size=1180,720
elif [ -d "/Applications/Brave Browser.app" ]; then
  open -na "Brave Browser" --args --app="${URL}" --window-size=1180,720
elif [ -d "/Applications/Microsoft Edge.app" ]; then
  open -na "Microsoft Edge" --args --app="${URL}" --window-size=1180,720
else
  open "${URL}"
fi
