#!/usr/bin/env bash
set -euo pipefail

# Watches an in-progress training process, then force-adds YOLO weights/results
# (even if gitignored), pushes to origin, and optionally shuts down the machine.
#
# Usage examples:
#   ./watch_push_shutdown.sh --pid 12345
#   ./watch_push_shutdown.sh --match "python ocr.py --train" --shutdown
#   ./watch_push_shutdown.sh --run-dir runs/detect/train13 --shutdown

PID=""
MATCH="python ocr.py --train"
INTERVAL_SECS=60
RUN_DIR=""
DO_SHUTDOWN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid)
      PID="$2"; shift 2 ;;
    --match)
      MATCH="$2"; shift 2 ;;
    --interval)
      INTERVAL_SECS="$2"; shift 2 ;;
    --run-dir)
      RUN_DIR="$2"; shift 2 ;;
    --shutdown)
      DO_SHUTDOWN=1; shift 1 ;;
    -h|--help)
      sed -n '1,120p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ -z "$PID" ]]; then
  # Pick the oldest matching PID if multiple are running.
  PID="$(pgrep -of -f "$MATCH" || true)"
fi

if [[ -n "$PID" ]]; then
  echo "Waiting for PID $PID to exit..."
  while kill -0 "$PID" 2>/dev/null; do
    sleep "$INTERVAL_SECS"
  done
  echo "PID $PID exited. Proceeding to push artifacts."
else
  echo "No PID provided and no process matched '$MATCH'. Proceeding immediately." >&2
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -1dt runs/detect/train*/ 2>/dev/null | head -n 1 || true)"
  RUN_DIR="${RUN_DIR%/}"
fi

if [[ -z "$RUN_DIR" ]]; then
  echo "Could not find a runs/detect/train*/ directory. Aborting." >&2
  exit 1
fi

WEIGHTS_DIR="$RUN_DIR/weights"
BEST_PT="$WEIGHTS_DIR/best.pt"
LAST_PT="$WEIGHTS_DIR/last.pt"
RESULTS_CSV="$RUN_DIR/results.csv"

if [[ ! -f "$BEST_PT" && ! -f "$LAST_PT" ]]; then
  echo "No weights found at: $WEIGHTS_DIR (expected best.pt/last.pt). Aborting." >&2
  exit 1
fi

echo "Using run dir: $RUN_DIR"
ls -lh "$WEIGHTS_DIR" || true

# Stage artifacts (force add in case *.pt is gitignored)
if [[ -f "$BEST_PT" ]]; then
  git add -f "$BEST_PT"
fi
if [[ -f "$LAST_PT" ]]; then
  git add -f "$LAST_PT"
fi
if [[ -f "$RESULTS_CSV" ]]; then
  git add -f "$RESULTS_CSV"
fi

if git diff --cached --quiet; then
  echo "No new artifacts to commit."
else
  TS="$(date -u +%Y%m%dT%H%M%SZ)"
  git commit -m "Add training artifacts from ${RUN_DIR##*/} ($TS)"
fi

git push

echo "Push complete."

if [[ "$DO_SHUTDOWN" -eq 1 ]]; then
  echo "Shutting down..."
  if command -v sudo >/dev/null 2>&1; then
    sudo shutdown -h now
  else
    shutdown -h now
  fi
fi
