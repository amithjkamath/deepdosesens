# -*- coding: utf-8 -*-
"""Loading and slimming dose-predictor checkpoints.

The archived checkpoints were written by the original training loop as a pickle
holding the whole trainer state:

===========================  ==========  ===============================
Key                          Size        Needed for inference?
===========================  ==========  ===============================
``network_state_dict``       129 MB      yes -- these are the weights
``optimizer_state_dict``     259 MB      no  -- Adam's two moment buffers
``lr_scheduler_state_dict``  < 1 KB      no
``log``                      < 1 MB      no  -- per-epoch loss history
===========================  ==========  ===============================

So a 518 MB file carries 129 MB of weights. :func:`slim` rewrites it as a
weights-only file, which is what :func:`load_state_dict` prefers. Slim files are
plain tensor dicts, so they load under ``weights_only=True`` and need none of
the legacy-module handling below.

Unpickling an *original* checkpoint requires the module layout of the 2022
training code (``training.network_trainer`` etc.), which no longer exists here.
:class:`_LegacyModule` stands in for it: the trainer object is only reachable
from the ``log`` entry, which inference never touches.
"""

import sys
import types
from pathlib import Path

import torch

# The cascaded 3D U-Net used for every result in both papers. in_ch = 15 is the
# CT plus the target volume plus 13 OAR masks.
DOSE_PREDICTOR_ARCH = dict(
    in_ch=15,
    out_ch=1,
    list_ch_A=[-1, 16, 32, 64, 128, 256],
    list_ch_B=[-1, 32, 64, 128, 256, 512],
)

# Everything in a full checkpoint that is training state rather than weights.
TRAINING_ONLY_KEYS = ("optimizer_state_dict", "lr_scheduler_state_dict", "log")

_LEGACY_MODULE_NAMES = (
    "training",
    "training.network_trainer",
    "training.trainer",
    "model",
    "model.model",
    "online_evaluation",
    "data",
)


class _LegacyModule(types.ModuleType):
    """Stand-in for the 2022 training package the old pickles reference.

    Any attribute resolves to a fresh empty class, so the unpickler can rebuild
    the trainer's bookkeeping objects without their original definitions.
    """

    __path__ = []  # marks this as a package so submodule imports resolve

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return type(attr, (object,), {})


def _install_legacy_modules():
    for name in _LEGACY_MODULE_NAMES:
        sys.modules.setdefault(name, _LegacyModule(name))


def load_state_dict(path):
    """Read network weights from either a slim or an original checkpoint."""
    path = Path(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # An original trainer checkpoint: pickled custom objects, so the legacy
        # module layout has to be in place and weights_only cannot be used.
        _install_legacy_modules()
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "network_state_dict" in obj:
        return obj["network_state_dict"]
    return obj


def load_dose_predictor(path, device="cpu"):
    """Build the cascaded 3D U-Net and load `path` into it, ready to predict."""
    from deepdosesens.model.model import CascadedUNet

    network = CascadedUNet(**DOSE_PREDICTOR_ARCH)
    network.load_state_dict(load_state_dict(path), strict=True)
    return network.to(torch.device(device)).eval()


def slim(src, dst):
    """Rewrite a checkpoint as weights only.

    Returns ``(src_bytes, dst_bytes)``. Bit-exact for inference: the tensors are
    copied unchanged, only the training state around them is dropped.
    """
    src, dst = Path(src), Path(dst)
    state = load_state_dict(src)
    state = {k: v.detach().clone() for k, v in state.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dst)
    return src.stat().st_size, dst.stat().st_size
