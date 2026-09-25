#!/usr/bin/env bash
# Verify a re-downloaded CIC-IDS-2018 capture day.
#
#   ./verify_pcap_day.sh wed_14_pcap
#
# Walks every capture's record chain with capinfos and compares the result
# against the 2026-09-19 baseline in corrupt_files_redownload.txt.
#
# Needs wireshark-cli. Run it inside the toolbox that has it:
#   toolbox run -c prism-dev ./verify_pcap_day.sh wed_14_pcap

set -uo pipefail

DAY="${1:-}"
ROOT="${PCAP_ROOT:-$HOME/Documents/SIH/DATA/pcap}"
BASELINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/corrupt_files_redownload.txt"

if [ -z "$DAY" ]; then
    echo "usage: $0 <day-dir>    e.g. $0 wed_14_pcap" >&2
    echo "days:  $(ls "$ROOT" 2>/dev/null | tr '\n' ' ')" >&2
    exit 2
fi
[ -d "$ROOT/$DAY" ] || { echo "no such day: $ROOT/$DAY" >&2; exit 2; }
command -v capinfos >/dev/null || {
    echo "capinfos not found — run this inside the toolbox:" >&2
    echo "  toolbox run -c prism-dev $0 $DAY" >&2
    exit 2
}

cd "$ROOT" || exit 2
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "Scanning $DAY ..."

check() {
    f="$1"
    out=$(capinfos -c "$f" 2>&1)
    if grep -q "bigger than maximum\|not a capture file" <<<"$out"; then
        echo "SEVERE|$f"
    elif grep -q "cut short in the middle of a packet" <<<"$out"; then
        echo "MINOR|$f"      # tail-only loss, coverage intact — not a defect worth refetching
    elif [ -z "$out" ]; then
        echo "SEVERE|$f"
    else
        echo "OK|$f"
    fi
}
export -f check

find "./$DAY" -type f ! -name '*.lnk' -print0 \
    | xargs -0 -P 6 -I{} bash -c 'check "$@"' _ {} > "$TMP/now.tsv"

total=$(wc -l < "$TMP/now.tsv")
grep '^SEVERE|' "$TMP/now.tsv" | cut -d'|' -f2 | sed 's|^\./||' | sort > "$TMP/severe.txt"
sev=$(wc -l < "$TMP/severe.txt")
minor=$(grep -c '^MINOR|' "$TMP/now.tsv")

echo
echo "  files scanned : $total"
echo "  SEVERE        : $sev   (record-chain corruption — real coverage loss)"
echo "  MINOR         : $minor   (truncated tail — coverage intact, ignore)"
echo

if [ ! -f "$BASELINE" ]; then
    echo "No baseline found at $BASELINE — reporting current state only."
    [ "$sev" -gt 0 ] && { echo; echo "Still corrupt:"; sed 's/^/  /' "$TMP/severe.txt"; }
    exit $(( sev > 0 ))
fi

grep "^$DAY/" "$BASELINE" | sort > "$TMP/was.txt"
was=$(wc -l < "$TMP/was.txt")
fixed=$(comm -23 "$TMP/was.txt" "$TMP/severe.txt" | wc -l)
still=$(comm -12 "$TMP/was.txt" "$TMP/severe.txt" | wc -l)
new=$(comm -13 "$TMP/was.txt" "$TMP/severe.txt" | wc -l)

echo "  vs baseline ($was severe before):"
echo "    fixed         : $fixed"
echo "    still corrupt : $still"
echo "    newly corrupt : $new"
echo

if [ "$still" -gt 0 ]; then
    echo "  Still corrupt after re-download:"
    comm -12 "$TMP/was.txt" "$TMP/severe.txt" | sed 's/^/    /'
    echo
    if [ "$fixed" -eq 0 ]; then
        echo "  >> Nothing was fixed. The damage is upstream, not in your transfer."
        echo "     Re-downloading other days will not help. Use tshark prefix"
        echo "     ingestion instead — every corrupt file has a readable prefix."
    fi
fi
if [ "$new" -gt 0 ]; then
    echo "  NEWLY corrupt (were fine before — suspect this transfer):"
    comm -13 "$TMP/was.txt" "$TMP/severe.txt" | sed 's/^/    /'
fi
[ "$sev" -eq 0 ] && echo "  Clean. No severe corruption in $DAY."

exit $(( sev > 0 ))
