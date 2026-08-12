#!/usr/bin/env bash
# Rebuild every demonstration video.
#
# Requires the unpacked cases and predictions (scripts/fetch_artifacts.sh) plus
# ffmpeg on PATH. Archive-quality encodes go to results/videos; smaller copies for
# the repository go to docs/videos.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

ARCHIVE_CRF="${ARCHIVE_CRF:-20}"
REPO_CRF="${REPO_CRF:-30}"
FPS="${FPS:-10}"

# Three dose sweeps over demanding test cases, the optic nerve sensitivity clip,
# and the two Cancers robustness comparisons.
# The default selectors pick by target size, organ-at-risk involvement and streak
# reproduction, and print the ranking of all 20 test cases first.
"$PY" -m deepdosesens.visualization.make_dose_video \
      --fps "$FPS" --crf "$ARCHIVE_CRF" --keep-frames
"$PY" -m deepdosesens.visualization.make_sensitivity_video \
      --fps "$FPS" --crf "$ARCHIVE_CRF" --keep-frames
"$PY" -m deepdosesens.visualization.make_robustness_video \
      --scenario all --fps "$FPS" --crf "$ARCHIVE_CRF" --keep-frames

# The repo copies are a second encode of the same frames: no re-rendering.
mkdir -p docs/videos
reencode() { # frames_dir  out_file
  ffmpeg -y -loglevel error \
    -framerate "$FPS" -i "$1/frame_%04d.png" \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    -c:v libx264 -pix_fmt yuv420p -crf "$REPO_CRF" \
    -tune stillimage -movflags +faststart "$2"
}
for frames in results/videos/frames_*; do
  name="$(basename "$frames")"; name="${name#frames_}"
  case "$name" in
    sensitivity) out="docs/videos/optic_nerve_sensitivity.mp4" ;;
    concave|multiple) out="docs/videos/robustness_$name.mp4" ;;
    *) out="docs/videos/dose_sweep_$name.mp4" ;;
  esac
  reencode "$frames" "$out"
done

# Frames are large and fully regenerable.
rm -rf results/videos/frames_*

echo
echo "repo copies:"
ls -la docs/videos/*.mp4
echo "archive copies:"
ls -la results/videos/*.mp4
