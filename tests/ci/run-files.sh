#!/usr/bin/env bash
# Run each test file listed in $1 in its own pytest process, as CI does, and fail if any file fails.
set -u
status=0
while IFS= read -r line; do
  file="${line%%#*}"
  file="${file//[[:space:]]/}"
  [ -z "$file" ] && continue
  echo "==> $file"
  uv run --offline --no-sync --extra dev pytest -q -p no:cacheprovider "$file" || status=1
done < "$1"
exit "$status"
