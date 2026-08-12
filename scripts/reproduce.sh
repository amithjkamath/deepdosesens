#!/usr/bin/env bash
# Recompute both papers' reported numbers from the archived volumes and print, for
# each claim, the published value beside the reproduced one.
#
# Requires the unpacked cases and predictions (scripts/fetch_artifacts.sh).
# Writes tables and a verification report per paper into the results directory.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

"$PY" -m deepdosesens.analyze.reproduce_isbi_tables
echo
"$PY" -m deepdosesens.analyze.reproduce_cancers_tables

echo
echo "verification reports:"
ls -la results/*_verification.csv
