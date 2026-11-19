#!/usr/bin/env bash
# Delete every folder under databases/ whose name contains "archive" (archive_other_runs,
# archive_short_runs, archive_pre_lns, ...) with all its contents.
# Default is a dry run that only lists them; pass --delete to actually remove.
set -euo pipefail
cd "$(dirname "$0")"

mapfile -d '' dirs < <(find databases -type d -name '*archive*' -prune -print0)
[ ${#dirs[@]} -eq 0 ] && { echo "No archive folders."; exit 0; }

printf '%s\n' "${dirs[@]}"
du -ch "${dirs[@]}" | tail -1

if [ "${1:-}" = "--delete" ]; then
  rm -rf -- "${dirs[@]}"
  echo "Deleted ${#dirs[@]} folders."
else
  echo "Dry run: ${#dirs[@]} folders. Run 'bash remove_archives.sh --delete' to remove them."
fi
