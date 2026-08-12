#!/usr/bin/env bash
# Unpack the planning data, predictions and model weights from the artifact
# archives, then release the archive copies again so local disk is not left full.
#
# Configure with environment variables or a .env file in the repository root
# (see .env.example); nothing is hardcoded:
#
#   DEEPDOSESENS_ISBI_ARCHIVE     where the ISBI zip files are
#   DEEPDOSESENS_CANCERS_ARCHIVE  where the Cancers zip files are (optional)
#   DEEPDOSESENS_DATA             where to unpack cases        (default <repo>/data)
#   DEEPDOSESENS_PREDICTIONS      where to unpack predictions  (default <repo>/predictions)
#   DEEPDOSESENS_CHECKPOINTS      where to unpack weights      (default <repo>/checkpoints)
#
# Example:
#   DEEPDOSESENS_ISBI_ARCHIVE=/Volumes/Share/2022-11-ISBI/artifacts scripts/fetch_artifacts.sh
#
# WHAT selects how much to pull:
#   WHAT=isbi      the ISBI paper's data, predictions and weights   (~1.2 GB)
#   WHAT=cancers   the Cancers paper's extra cases and models       (~1.5 GB)
#   WHAT=all       both (default)
#
# Each archive carries a README.md manifest describing its layout and contents.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

# Single source of truth for every location: deepdosesens/config.py.
eval "$("$PY" - <<'EOF'
from deepdosesens import config
print(f'ISBI_ARCHIVE="{config.ISBI_ARCHIVE_DIR or ""}"')
print(f'CANCERS_ARCHIVE="{config.CANCERS_ARCHIVE_DIR or ""}"')
print(f'DATA_DIR="{config.DATA_DIR}"')
print(f'PRED_DIR="{config.PREDICTIONS_DIR}"')
print(f'CKPT_DIR="{config.CHECKPOINTS_DIR}"')
EOF
)"

WHAT="${WHAT:-all}"
case "$WHAT" in
  all|isbi|cancers) ;;
  *) echo "error: WHAT must be one of: all, isbi, cancers" >&2; exit 1 ;;
esac

echo "ISBI archive     ${ISBI_ARCHIVE:-(unset)}"
echo "Cancers archive  ${CANCERS_ARCHIVE:-(unset)}"
echo "data             $DATA_DIR"
echo "predictions      $PRED_DIR"
echo "checkpoints      $CKPT_DIR"
echo

if [[ "$WHAT" != "cancers" && -z "$ISBI_ARCHIVE" ]]; then
  echo "error: no ISBI artifact archive configured." >&2
  echo "  set DEEPDOSESENS_ISBI_ARCHIVE, e.g." >&2
  echo "    DEEPDOSESENS_ISBI_ARCHIVE=~/Documents/.../2022-11-ISBI/artifacts $0" >&2
  echo "  or copy .env.example to .env and edit it." >&2
  exit 1
fi
if [[ "$WHAT" != "isbi" && -z "$CANCERS_ARCHIVE" ]]; then
  echo "error: no Cancers artifact archive configured; set DEEPDOSESENS_CANCERS_ARCHIVE" >&2
  echo "  or run with WHAT=isbi to fetch only the ISBI artifacts." >&2
  exit 1
fi

# Files synced by a cloud provider may be placeholders; materialise on demand and
# release afterwards. Harmless no-ops for a plain local directory.
ON_CLOUD=0
command -v brctl >/dev/null 2>&1 && ON_CLOUD=1

materialise() {
  local f="$1"
  [[ -f "$f" ]] || { echo "error: missing archive file: $f" >&2; exit 1; }
  [[ $ON_CLOUD -eq 1 ]] || return 0
  ls -lO "$f" 2>/dev/null | grep -q dataless || return 0
  echo "  downloading $(basename "$f") ..."
  brctl download "$f"
  for _ in $(seq 1 360); do
    ls -lO "$f" | grep -q dataless || return 0
    sleep 5
  done
  echo "error: timed out downloading $f" >&2
  exit 1
}

release() {
  [[ $ON_CLOUD -eq 1 ]] || return 0
  brctl evict "$1" >/dev/null 2>&1 || true
}

# extract <zip> <dest> [strip_prefix] [rename_map_json]
#   strip_prefix     leading path component to drop
#   rename_map_json  {"archive/dir": "local-dir"} applied to the first component
#                    after stripping, so the local tree carries descriptive names
extract() {
  ZIP="$1" DEST="$2" STRIP="${3:-}" RENAME="${4:-{\}}" "$PY" - <<'EOF'
import json, os, zipfile

zip_path, dest = os.environ["ZIP"], os.environ["DEST"]
strip, rename = os.environ["STRIP"], json.loads(os.environ["RENAME"])
archive = zipfile.ZipFile(zip_path)
written = 0
for info in archive.infolist():
    name = info.filename
    if name.endswith("/") or name.startswith("__MACOSX") or ".DS_Store" in name:
        continue
    parts = name.split("/")
    if strip and parts and parts[0] == strip:
        parts = parts[1:]
    if not parts:
        continue
    # Longest-prefix rename, so both "a/b" and "a" can be remapped.
    for source, target in sorted(rename.items(), key=lambda kv: -len(kv[0])):
        source_parts = source.split("/")
        if parts[: len(source_parts)] == source_parts:
            parts = target.split("/") + parts[len(source_parts):]
            break
    target_path = os.path.join(dest, *parts)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with archive.open(info) as src, open(target_path, "wb") as out:
        out.write(src.read())
    written += 1
print(f"  {os.path.basename(zip_path)}: {written} files -> {dest}")
EOF
}

if [[ "$WHAT" != "cancers" ]]; then
  echo "==> ISBI: planning cases (100 glioblastoma cases, 10 optic nerve variants)"
  materialise "$ISBI_ARCHIVE/inputs/glioblastoma-cases.zip"
  extract "$ISBI_ARCHIVE/inputs/glioblastoma-cases.zip" "$DATA_DIR" "" \
          '{"processed-dldp": "glioblastoma"}'
  release "$ISBI_ARCHIVE/inputs/glioblastoma-cases.zip"

  materialise "$ISBI_ARCHIVE/inputs/optic-nerve-variant-cases.zip"
  extract "$ISBI_ARCHIVE/inputs/optic-nerve-variant-cases.zip" "$DATA_DIR" "" \
          '{"processed-ONL": "optic-nerve-variants"}'
  release "$ISBI_ARCHIVE/inputs/optic-nerve-variant-cases.zip"

  echo "==> ISBI: predicted dose volumes for six training runs"
  # The zips keep their original internal directory names; the rename map turns
  # them into the version-free local layout. See the archive README.
  materialise "$ISBI_ARCHIVE/predictions/glioblastoma.zip"
  extract "$ISBI_ARCHIVE/predictions/glioblastoma.zip" "$PRED_DIR/glioblastoma" "" "$(
    "$PY" -c '
import json
runs = {f"output-dldp-{i}/Prediction": f"run-{i}" for i in range(1, 7)}
runs.update({f"output-dldp-dt-{i}/Prediction": f"distance-transform-run-{i}" for i in range(1, 4)})
runs.update({f"output-dldp-{i}/log.txt": f"run-{i}/source-checkpoint.txt" for i in range(1, 7)})
runs.update({f"output-dldp-dt-{i}/log.txt": f"distance-transform-run-{i}/source-checkpoint.txt"
             for i in range(1, 4)})
print(json.dumps(runs))'
  )"
  release "$ISBI_ARCHIVE/predictions/glioblastoma.zip"

  materialise "$ISBI_ARCHIVE/predictions/optic-nerve-variants.zip"
  extract "$ISBI_ARCHIVE/predictions/optic-nerve-variants.zip" \
          "$PRED_DIR/optic-nerve-variants" "" "$(
    "$PY" -c '
import json
runs = {f"output-ONL-{i}/Prediction": f"run-{i}" for i in range(1, 7)}
runs.update({f"output-ONL-dt-{i}/Prediction": f"distance-transform-run-{i}" for i in range(1, 4)})
runs.update({f"output-ONL-{i}/log.txt": f"run-{i}/source-checkpoint.txt" for i in range(1, 7)})
runs.update({f"output-ONL-dt-{i}/log.txt": f"distance-transform-run-{i}/source-checkpoint.txt"
             for i in range(1, 4)})
print(json.dumps(runs))'
  )"
  release "$ISBI_ARCHIVE/predictions/optic-nerve-variants.zip"

  echo "==> ISBI: dose-predictor weights"
  materialise "$ISBI_ARCHIVE/checkpoints/dose-predictor-weights.zip"
  extract "$ISBI_ARCHIVE/checkpoints/dose-predictor-weights.zip" "$CKPT_DIR"
  release "$ISBI_ARCHIVE/checkpoints/dose-predictor-weights.zip"
fi

if [[ "$WHAT" != "isbi" ]]; then
  echo "==> Cancers: 120 planning cases, including the out-of-distribution targets"
  materialise "$CANCERS_ARCHIVE/inputs/glioblastoma-cases.zip"
  extract "$CANCERS_ARCHIVE/inputs/glioblastoma-cases.zip" "$DATA_DIR" "" \
          '{"processed-dldp": "glioblastoma"}'
  release "$CANCERS_ARCHIVE/inputs/glioblastoma-cases.zip"

  echo "==> Cancers: predicted dose for the initial and three retrained models"
  # These zips already carry descriptive internal names, so no rename is needed.
  materialise "$CANCERS_ARCHIVE/predictions/glioblastoma.zip"
  extract "$CANCERS_ARCHIVE/predictions/glioblastoma.zip" "$PRED_DIR/glioblastoma"
  release "$CANCERS_ARCHIVE/predictions/glioblastoma.zip"

  materialise "$CANCERS_ARCHIVE/predictions/optic-nerve-variants.zip"
  extract "$CANCERS_ARCHIVE/predictions/optic-nerve-variants.zip" \
          "$PRED_DIR/optic-nerve-variants"
  release "$CANCERS_ARCHIVE/predictions/optic-nerve-variants.zip"

  echo "==> Cancers: weights for all four models"
  materialise "$CANCERS_ARCHIVE/checkpoints/dose-predictor-weights.zip"
  extract "$CANCERS_ARCHIVE/checkpoints/dose-predictor-weights.zip" "$CKPT_DIR"
  release "$CANCERS_ARCHIVE/checkpoints/dose-predictor-weights.zip"
fi

echo
echo "done."
du -sh "$DATA_DIR" "$PRED_DIR" "$CKPT_DIR" 2>/dev/null || true
echo
"$PY" -m deepdosesens.config
