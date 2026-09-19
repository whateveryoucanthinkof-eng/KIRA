#!/usr/bin/env bash
# Re-fetch only the 263 corrupt CIC-IDS-2018 captures (56.2 GB) instead of all 559 GB.
#
# Uses `aws s3 cp`, which verifies each part's checksum. That matters: the corruption
# in the current copy is a multi-connection download artefact (0% on files <10 MB,
# 16.7% on 200-500 MB files), not damage in the published dataset.
#
#   ./refetch_corrupt.sh --dry-run          # show what would be fetched
#   ./refetch_corrupt.sh --day wed_14_pcap  # one day at a time
#   ./refetch_corrupt.sh                    # everything
#
# BEFORE FIRST USE: set S3_PREFIX below to the real bucket path, and confirm it with
#   aws s3 ls --no-sign-request s3://cse-cic-ids2018/
# The local layout is <day>/pcap/<capture>; the bucket's layout may differ, so map
# LOCAL_DAY -> S3 path in day_to_s3() once and the rest follows.

set -uo pipefail

S3_PREFIX="${S3_PREFIX:-s3://cse-cic-ids2018}"
DEST_ROOT="${PCAP_ROOT:-$HOME/Documents/SIH/DATA/pcap}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="$HERE/corrupt_files_redownload.txt"

DRY=0; ONLY_DAY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --day) ONLY_DAY="${2:-}"; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

[ -f "$LIST" ] || { echo "missing $LIST" >&2; exit 2; }
if [ "$DRY" -eq 0 ] && ! command -v aws >/dev/null; then
    echo "aws cli not found. Install it in the toolbox:" >&2
    echo "  toolbox run -c prism-dev sudo dnf install -y awscli2" >&2
    echo "then re-run this inside it:" >&2
    echo "  toolbox run -c prism-dev $0 ${ONLY_DAY:+--day $ONLY_DAY}" >&2
    exit 2
fi

# Local day dir -> path inside the bucket. ADJUST AFTER CONFIRMING THE BUCKET LAYOUT.
day_to_s3() {
    case "$1" in
        wed_14_pcap)  echo "Original Network Traffic and Log data/Wednesday-14-02-2018" ;;
        thu_15_pacap) echo "Original Network Traffic and Log data/Thursday-15-02-2018" ;;
        fri_16_pcap)  echo "Original Network Traffic and Log data/Friday-16-02-2018" ;;
        tue_20_pcap)  echo "Original Network Traffic and Log data/Tuesday-20-02-2018" ;;
        wed_21_pcap)  echo "Original Network Traffic and Log data/Wednesday-21-02-2018" ;;
        thu_22_pcap)  echo "Original Network Traffic and Log data/Thursday-22-02-2018" ;;
        fri_23_pacap) echo "Original Network Traffic and Log data/Friday-23-02-2018" ;;
        wed_28_pcap)  echo "Original Network Traffic and Log data/Wednesday-28-02-2018" ;;
        thu_1_pcap)   echo "Original Network Traffic and Log data/Thursday-01-03-2018" ;;
        fri_2_pcap)   echo "Original Network Traffic and Log data/Friday-02-03-2018" ;;
        *) return 1 ;;
    esac
}

n=0; bytes=0; failed=0
while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    day="${rel%%/*}"
    [ -n "$ONLY_DAY" ] && [ "$day" != "$ONLY_DAY" ] && continue
    cap="${rel##*/}"

    s3day="$(day_to_s3 "$day")" || { echo "!! no S3 mapping for $day" >&2; failed=$((failed+1)); continue; }
    src="$S3_PREFIX/$s3day/pcap/$cap"
    dst="$DEST_ROOT/$rel"

    n=$((n+1))
    [ -f "$dst" ] && bytes=$((bytes + $(stat -c %s "$dst" 2>/dev/null || echo 0)))

    if [ "$DRY" -eq 1 ]; then
        printf '%s\n  -> %s\n' "$src" "$dst"
        continue
    fi

    # keep the damaged copy until the new one lands
    [ -f "$dst" ] && mv -f "$dst" "$dst.corrupt.bak"
    if aws s3 cp --no-sign-request "$src" "$dst" >/dev/null 2>&1; then
        rm -f "$dst.corrupt.bak"
        echo "ok      $rel"
    else
        [ -f "$dst.corrupt.bak" ] && mv -f "$dst.corrupt.bak" "$dst"
        echo "FAILED  $rel"
        failed=$((failed+1))
    fi
done < "$LIST"

echo
if [ "$DRY" -eq 1 ]; then
    printf 'dry run: %d files, %.1f GB (current sizes)\n' "$n" "$(echo "$bytes/1073741824" | bc -l)"
    echo "Confirm the bucket layout first:  aws s3 ls --no-sign-request $S3_PREFIX/"
else
    echo "fetched $((n-failed))/$n   failed: $failed"
    echo
    echo "Now verify:"
    if [ -n "$ONLY_DAY" ]; then
        echo "  toolbox run -c prism-dev $HERE/verify_pcap_day.sh $ONLY_DAY"
    else
        echo "  for d in \$(ls $DEST_ROOT); do toolbox run -c prism-dev $HERE/verify_pcap_day.sh \$d; done"
    fi
fi
exit $(( failed > 0 ))
