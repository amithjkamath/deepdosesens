# -*- coding: utf-8 -*-
"""Path configuration for the whole project.

Nothing in this repository hardcodes a machine-specific path. Every script
resolves its inputs and outputs through this module, which reads (in order of
precedence):

1. environment variables,
2. a ``.env`` file in the repository root,
3. defaults relative to the repository root.

Recognised settings -- see ``.env.example``:

=================================  ======================================  ==================
Variable                           Meaning                                 Default
=================================  ======================================  ==================
``DEEPDOSESENS_ISBI_ARCHIVE``      Zip archives for the ISBI paper          unset
``DEEPDOSESENS_CANCERS_ARCHIVE``   Zip archives for the Cancers paper       unset
``DEEPDOSESENS_DATA``              CT, contours and reference dose plans    ``<repo>/data``
``DEEPDOSESENS_CHECKPOINTS``       Trained dose-predictor weights           ``<repo>/checkpoints``
``DEEPDOSESENS_PREDICTIONS``       Predicted dose volumes per model         ``<repo>/predictions``
``DEEPDOSESENS_RESULTS``           Tables, figures and videos               ``<repo>/results``
=================================  ======================================  ==================

So a collaborator who receives the zip files only has to write, for example::

    DEEPDOSESENS_ISBI_ARCHIVE=/Volumes/Shared/2022-11-ISBI/artifacts

and run ``scripts/fetch_artifacts.sh``; or, if they unpacked the archives
somewhere else already::

    DEEPDOSESENS_DATA=/mnt/big-disk/deepdosesens/data
    DEEPDOSESENS_CHECKPOINTS=/mnt/big-disk/deepdosesens/checkpoints

Usage::

    from deepdosesens.config import data_path, prediction_path

    case = data_path("glioblastoma", "DLDP_081")
    dose = prediction_path("glioblastoma", "run-6", "DLDP_081", "Dose.nii.gz")
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:  # optional, and only used to populate os.environ
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover - keep usable without python-dotenv
    pass


def _resolve(var, default):
    value = os.environ.get(var)
    if not value:
        return default
    return Path(value).expanduser().resolve()


DATA_DIR = _resolve("DEEPDOSESENS_DATA", REPO_ROOT / "data")
CHECKPOINTS_DIR = _resolve("DEEPDOSESENS_CHECKPOINTS", REPO_ROOT / "checkpoints")
PREDICTIONS_DIR = _resolve("DEEPDOSESENS_PREDICTIONS", REPO_ROOT / "predictions")
RESULTS_DIR = _resolve("DEEPDOSESENS_RESULTS", REPO_ROOT / "results")

# Only needed by scripts/fetch_artifacts.sh; None when unset.
ISBI_ARCHIVE_DIR = _resolve("DEEPDOSESENS_ISBI_ARCHIVE", None)
CANCERS_ARCHIVE_DIR = _resolve("DEEPDOSESENS_CANCERS_ARCHIVE", None)


def data_path(*parts):
    """Path under the data directory."""
    return DATA_DIR.joinpath(*parts)


def checkpoint_path(*parts):
    """Path under the checkpoints directory."""
    return CHECKPOINTS_DIR.joinpath(*parts)


def prediction_path(*parts):
    """Path under the predictions directory."""
    return PREDICTIONS_DIR.joinpath(*parts)


def results_path(*parts):
    """Path under the results directory."""
    return RESULTS_DIR.joinpath(*parts)


def require(path, hint="run scripts/fetch_artifacts.sh to unpack the artifacts"):
    """Fail with an actionable message instead of an obscure empty-glob error."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing required path: {p}\n  {hint}")
    return p


def describe():
    """One line per configured location, for logs and troubleshooting."""
    return "\n".join(
        [
            f"repo         {REPO_ROOT}",
            f"data         {DATA_DIR}",
            f"checkpoints  {CHECKPOINTS_DIR}",
            f"predictions  {PREDICTIONS_DIR}",
            f"results      {RESULTS_DIR}",
            f"ISBI archive     {ISBI_ARCHIVE_DIR if ISBI_ARCHIVE_DIR else '(unset)'}",
            f"Cancers archive  {CANCERS_ARCHIVE_DIR if CANCERS_ARCHIVE_DIR else '(unset)'}",
        ]
    )


if __name__ == "__main__":
    print(describe())
