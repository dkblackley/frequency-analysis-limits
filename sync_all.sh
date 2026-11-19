#!/usr/bin/env bash
# Pull LAMa (limits/), Even Less and Remin/Remin+ results from hopper into databases/.
# Only grids: 50x50 (merged), 50x50_run*, 16x16x14 (nh), 6x6 ... 6x6x6x6x6x6.
# Skips archive_* folders (archive_other_runs, archive_short_runs, archive_pre_lns).
# Never deletes local files. Extra args go to rsync, e.g. `bash sync_all.sh --dry-run`.
set -euo pipefail
cd "$(dirname "$0")"

rsync -av --prune-empty-dirs "$@" \
  --include='/50x50/' --include='/50x50_run*/' --include='/16x16x14/' --include='/6x6*/' \
  --exclude='/*' \
  --exclude='*archive*/' \
  --include='*/' \
  --include='*/limits/**' \
  --include='*/even_less/**' --include='*/remin/**' --include='*/reminp/**' \
  --exclude='*' \
  hopper:/scratch/dblackle/frequency-analysis-limits/databases/ databases/
