# -*- coding: utf-8 -*-
"""Shared drawing primitives for the demonstration videos.

One dose panel is a planning CT in grey, a dose heat map over it, and the target
volume and organs at risk as outlines. Every video in this package is built from
that panel, so the colours, window and opacity ramp live here and are identical
across clips.
"""

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib import patheffects
from matplotlib.lines import Line2D

# One hue per organ, from a palette validated for colour-vision deficiency
# (scripts/validate_palette.js in the dataviz skill: worst adjacent CVD dE 9.1).
# Left and right instances of an organ share a hue -- which side is which is never
# ambiguous, since the image itself places them either side of the midline. The
# target volume is not an organ, so it takes white and the heaviest line: it is
# what the plan is built around.
STRUCTURE_COLOURS = {
    "Target": "#ffffff",
    "BrainStem": "#2a78d6",
    "Chiasm": "#eb6834",
    "Cochlea": "#1baf7a",
    "Eye": "#eda100",
    "Hippocampus": "#e87ba4",
    "LacrimalGland": "#008300",
    "OpticNerve": "#4a3aa7",
    "Pituitary": "#e34948",
}
STRUCTURE_LABELS = {
    "Target": "Target volume (PTV)",
    "BrainStem": "Brainstem",
    "Chiasm": "Chiasm",
    "Cochlea": "Cochlea (L, R)",
    "Eye": "Eye (L, R)",
    "Hippocampus": "Hippocampus (L, R)",
    "LacrimalGland": "Lacrimal gland (L, R)",
    "OpticNerve": "Optic nerve (L, R)",
    "Pituitary": "Pituitary",
}
# Files on disk, grouped onto the organ that owns them.
STRUCTURE_FILES = {
    "Target": ["Target"],
    "BrainStem": ["BrainStem"],
    "Chiasm": ["Chiasm"],
    "Cochlea": ["Cochlea_L", "Cochlea_R"],
    "Eye": ["Eye_L", "Eye_R"],
    "Hippocampus": ["Hippocampus_L", "Hippocampus_R"],
    "LacrimalGland": ["LacrimalGland_L", "LacrimalGland_R"],
    "OpticNerve": ["OpticNerve_L", "OpticNerve_R"],
    "Pituitary": ["Pituitary"],
}

# Brain window on the planning CT: level 40, width 400 HU.
CT_WINDOW = (-160, 240)
# Perceptually uniform, and hot = high dose as clinicians read it.
DOSE_CMAP = "inferno"

INK = "#f2f2f0"
BACKGROUND = "#101014"


def outline_effect(linewidth=2.6):
    """A pale halo, so a contour stays legible over dark and bright dose alike."""
    return [patheffects.withStroke(linewidth=linewidth, foreground="white", alpha=0.5)]


def read(path, dtype=sitk.sitkFloat32):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path), dtype))


def load_structures(case_dir):
    """Binary masks per organ, left and right merged."""
    masks = {}
    for organ, files in STRUCTURE_FILES.items():
        present = [
            f for f in files if os.path.exists(os.path.join(str(case_dir), f + ".nii.gz"))
        ]
        if present:
            masks[organ] = np.maximum.reduce(
                [read(os.path.join(str(case_dir), f + ".nii.gz")) for f in present]
            )
    return masks


def crop_box(body, margin=4):
    """One in-plane bounding box for a whole sweep, so the head fills the panel.

    Fixed across slices: a box that moved from frame to frame would read as the
    patient shifting.
    """
    occupied = body.sum(axis=0) > 0
    rows = np.flatnonzero(occupied.any(axis=1))
    cols = np.flatnonzero(occupied.any(axis=0))
    return (
        (max(0, rows.min() - margin), min(body.shape[1], rows.max() + margin + 1)),
        (max(0, cols.min() - margin), min(body.shape[2], cols.max() + margin + 1)),
    )


def slice_range(body, pad=2):
    """Slices that hold anatomy, in caudal-to-cranial order.

    The archived volumes were resampled to 128^3 with their orientation reset, so
    the direction matrix says nothing. Anatomy settles it: the eyes, pituitary and
    lower brainstem sit at low indices and the brain reaches its vertex at high
    ones, i.e. increasing index is cranial.
    """
    occupied = np.flatnonzero(body.reshape(body.shape[0], -1).sum(axis=1) > 0)
    lo = max(0, occupied.min() - pad)
    hi = min(body.shape[0] - 1, occupied.max() + pad)
    return list(range(lo, hi + 1))


def dose_overlay_rgba(dose_slice, dose_min, dose_max):
    """Colour the dose, and let opacity rise with it.

    A flat alpha hides the CT everywhere the dose is non-trivial. Ramping opacity
    with dose keeps the anatomy readable in the low-dose wash while the high-dose
    region, which is what the eye should go to, stays close to opaque.
    """
    normalised = np.clip((dose_slice - dose_min) / float(dose_max - dose_min), 0.0, 1.0)
    rgba = plt.get_cmap(DOSE_CMAP)(normalised)
    rgba[..., 3] = np.where(dose_slice < dose_min, 0.0, 0.32 + 0.46 * normalised)
    return rgba


def draw_panel(
    ax, ct_slice, dose_slice, masks_slice, title, dose_min, dose_max, extra_contours=()
):
    """CT + dose heat map + structure outlines in one axes.

    ``extra_contours`` takes ``(mask, colour, linewidth, linestyle)`` tuples, for
    overlays a panel needs beyond the case's own structures.
    """
    ax.imshow(
        ct_slice, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], interpolation="bilinear"
    )
    if dose_slice is not None:
        ax.imshow(dose_overlay_rgba(dose_slice, dose_min, dose_max), interpolation="bilinear")
    for organ, mask in masks_slice.items():
        if mask.max() == 0:
            continue
        contours = ax.contour(
            mask,
            levels=[0.5],
            colors=[STRUCTURE_COLOURS[organ]],
            linewidths=2.0 if organ == "Target" else 1.3,
        )
        contours.set_path_effects(outline_effect())
    for mask, colour, linewidth, linestyle in extra_contours:
        if mask.max() == 0:
            continue
        contours = ax.contour(
            mask, levels=[0.5], colors=[colour], linewidths=linewidth, linestyles=linestyle
        )
        contours.set_path_effects(outline_effect(linewidth + 1.3))
    if title:
        ax.set_title(title, color=INK, fontsize=12, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def structure_legend(ax, organs, title="Contoured structures", fontsize=9.5, extra=()):
    """Legend keyed by organ colour, plus any extra pre-built handles."""
    handles = [
        Line2D(
            [0], [0], color=STRUCTURE_COLOURS[organ],
            linewidth=3.0 if organ == "Target" else 2.2,
            label=STRUCTURE_LABELS[organ],
        )
        for organ in STRUCTURE_COLOURS
        if organ in organs
    ] + list(extra)
    legend = ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
        fontsize=fontsize, labelspacing=0.7, handlelength=1.8, title=title,
    )
    for text in legend.get_texts():
        text.set_color(INK)
    legend.get_title().set_color(INK)
    legend.get_title().set_fontsize(fontsize + 1)
    return legend


def dose_colorbar(fig, rect, dose_min, dose_max):
    """Vertical dose scale, with a note about the transparent floor."""
    bar_ax = fig.add_axes(rect)
    mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=dose_min, vmax=dose_max), cmap=DOSE_CMAP
    )
    colorbar = fig.colorbar(mappable, cax=bar_ax)
    colorbar.set_label("Dose (Gy)", color=INK, fontsize=10)
    colorbar.ax.yaxis.set_tick_params(color=INK, labelcolor=INK, labelsize=9)
    colorbar.outline.set_edgecolor(INK)
    colorbar.ax.text(
        0.5, -0.10, f"< {dose_min:g} Gy\nnot shaded", transform=colorbar.ax.transAxes,
        color=INK, fontsize=8.5, ha="center", va="top",
    )
    return colorbar


def encode(frame_dir, out_file, fps, crf=20):
    """Frames -> H.264 mp4, padded to even dimensions."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(frame_dir, "frame_%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
            "-movflags", "+faststart", out_file,
        ],
        check=True,
    )
