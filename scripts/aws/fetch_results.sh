#!/usr/bin/env bash
# Pull a completed job's results from S3 + print a summary.
#
# Usage:
#   ./scripts/aws/fetch_results.sh <job-name>
#
# Equivalent to scripts/nautilus/fetch_repro_results.sh but for AWS.
# S3 storage is durable (no pod-deletion footgun), so this is purely a
# convenience wrapper.

set -euo pipefail

JOB_NAME="${1:?usage: fetch_results.sh <job-name>}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/.agents/aws_infra/ids.env"

LOCAL_DIR="results/${JOB_NAME}"
mkdir -p "$LOCAL_DIR"

echo "=== syncing s3://${S3_BUCKET}/${JOB_NAME}/ -> ${LOCAL_DIR}/ ==="
aws s3 sync "s3://${S3_BUCKET}/${JOB_NAME}/" "$LOCAL_DIR/" --no-progress

echo
echo "=== contents ==="
ls -lh "$LOCAL_DIR"
echo

if [[ -f "$LOCAL_DIR/exit_code.txt" ]]; then
  EXIT=$(cat "$LOCAL_DIR/exit_code.txt")
  echo "exit_code: $EXIT"
fi

if [[ -f "$LOCAL_DIR/start.txt" ]]; then
  cat "$LOCAL_DIR/start.txt"
fi
if [[ -f "$LOCAL_DIR/end.txt" ]]; then
  cat "$LOCAL_DIR/end.txt"
fi

echo
echo "Last 30 lines of job.log:"
tail -30 "$LOCAL_DIR/job.log" 2>/dev/null || echo "(no job.log yet)"

echo
# Auto-detect known summary JSON shapes
for cand in eval_matrix.json reproduction_summary.json variance_summary.json; do
  if [[ -f "$LOCAL_DIR/$cand" ]]; then
    echo "=== ${cand} ==="
    # Use awk instead of `head -80` so the producer (`python3 -m json.tool`)
    # never sees SIGPIPE on long JSON. With `set -euo pipefail`, head closing
    # the pipe early would abort the whole script even though the sync
    # already succeeded -- making fetches flaky for any summary >80 lines
    # (e.g. eval_matrix.json). awk reads stdin to EOF, prints only the
    # first 80 lines, exits clean. Fix per Codex review on PR #2.
    python3 -m json.tool "$LOCAL_DIR/$cand" | awk 'NR<=80'
    break
  fi
done
