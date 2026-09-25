#!/usr/bin/env bash
# Run verify.py on each given day dir under a 4 GB memory cap; one JSON line per day.
# usage: run_verify.sh <label_dir> <day_dir>...
set -u
here="$(cd "$(dirname "$0")" && pwd)"
labels="$1"; shift
for d in "$@"; do
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- \
    nice python3 "$here/verify.py" "$d" "$labels" 2>/dev/null \
    | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)))"
done
