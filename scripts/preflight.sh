#!/usr/bin/env bash
# Run before launching any training. Catches, in seconds, the failures that
# otherwise surface 40 minutes in with the expensive work already paid for.
#
# Branch B died on 2026-09-22 with `NameError: name '_c' is not defined` --
# a trainer-local name used in main(). The check that needed it sits AFTER the
# trajectory stores are built, so extraction ran for 40 minutes and was then
# thrown away. ruff finds that class of bug statically in under a second.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY=/var/home/samito/.pyenv/versions/3.12.14/bin/python3
fail=0

echo "== 1. undefined names in live code (the NameError class) =="
LIVE="scripts data_unification branch_a_gnn_lstm branch_b_world_model deepop_decoder control_backend correlation cyberworld_v4"
if out=$($PY -m ruff check --select F821,F822 --no-cache --output-format=concise $LIVE 2>&1); then
    echo "   clean"
else
    echo "$out" | sed 's/^/   /'; fail=1
fi

echo "== 2. every trainer imports and its CLI parses =="
for s in scripts/retrain_branch_a_live.py scripts/retrain_future_models_live.py; do
    if $PY "$s" --help >/dev/null 2>&1; then echo "   $s ok"
    else echo "   $s FAILED --help"; $PY "$s" --help 2>&1 | tail -5 | sed 's/^/      /'; fail=1; fi
done

echo "== 3. contract is reachable from module scope =="
$PY - <<'PY' || fail=1
from cyberworld_v4.config import get_contract
c = get_contract()
assert c.window_seconds > 0 and c.history_steps > 0 and c.forecast_steps > 0
print(f"   {c.window_seconds}s window / {c.history_steps} history / {c.forecast_steps} forecast")
PY

echo "== 4. test suite =="
if systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --quiet \
     $PY -m pytest tests/ -q 2>&1 | tail -1 | tee /dev/stderr | grep -q "failed"; then
    fail=1
fi

echo "== 5. disk and memory headroom =="
avail_g=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
free_g=$(free -g | awk '/^Mem:/{print $7}')
echo "   disk ${avail_g}G free, RAM ${free_g}G available"
[ "$avail_g" -lt 20 ] && { echo "   REFUSE: under 20G disk"; fail=1; }
[ "$free_g" -lt 6 ]  && { echo "   REFUSE: under 6G RAM available"; fail=1; }

echo
[ "$fail" -eq 0 ] && echo "PREFLIGHT PASS -- safe to launch" || echo "PREFLIGHT FAIL -- do not launch"
exit $fail
