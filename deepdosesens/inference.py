# -*- coding: utf-8 -*-
"""Dose prediction inference.

The model is built and its weights read **once**, when a :class:`DosePredictor`
is constructed; every later call reuses the network already resident on the
device. Predicting a whole test set is therefore one load and N forward passes,
rather than N loads:

    predictor = DosePredictor(checkpoint_path("dose-predictor", "weights.pt"))
    for case in cases:
        predictor.predict_to_nifti(case, out_dir / case.name)

Command line:

    python -m deepdosesens.inference --cases glioblastoma --run dose-predictor
    python -m deepdosesens.inference --show-config
"""

import argparse
import os
import sys
import time

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepdosesens.config import (  # noqa: E402
    checkpoint_path,
    data_path,
    describe,
    prediction_path,
    require,
)
from deepdosesens.data.utils import (  # noqa: E402
    duplicate_image_metadata,
    flip_3d,
    pre_processing,
    read_data,
)
from deepdosesens.model.checkpoint import load_dose_predictor  # noqa: E402

# The predicted dose is learnt on [0, 1] and scaled to Gray for reporting; 70 Gy
# is the normalisation the training data used (60 Gy prescribed in 30 fractions).
DOSE_SCALE_GY = 70.0

# Test-time augmentation: flips over the axial (Z) and left-right (W) axes, whose
# predictions are averaged. This is what produced every archived prediction.
TTA_FLIPS = [[], ["Z"], ["W"], ["Z", "W"]]


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DosePredictor:
    """A loaded dose-prediction network, reusable across any number of cases."""

    def __init__(self, weights, device=None, tta=True):
        self.device = torch.device(device or default_device())
        self.weights = str(weights)
        self.network = load_dose_predictor(require(weights), self.device)
        self.tta = tta

    def _forward(self, volume):
        """One forward pass through the cascade; returns the refined output."""
        tensor = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))
        tensor = tensor.unsqueeze(0).to(self.device)
        # The cascade returns [coarse + refined, refined]; the refined output is
        # the prediction, as in the original training and evaluation code.
        output = self.network(tensor)[1]
        return np.array(output.cpu().data[0, :, :, :, :])

    def predict(self, case_dir):
        """Predicted dose in Gray for one case directory, shape (Z, H, W)."""
        require(case_dir)
        images = read_data(str(case_dir))
        network_input, possible_dose_mask = pre_processing(images)

        with torch.no_grad():
            flips = TTA_FLIPS if self.tta else [[]]
            predictions = [
                flip_3d(self._forward(flip_3d(network_input.copy(), axes)), axes)[0]
                for axes in flips
            ]
        prediction = np.mean(predictions, axis=0)

        # Dose only exists where the plan could deposit it, and never below zero.
        outside = np.logical_or(possible_dose_mask[0] < 1, prediction < 0)
        prediction[outside] = 0.0
        return DOSE_SCALE_GY * prediction

    def predict_to_nifti(self, case_dir, out_dir):
        """Predict one case and write ``Dose.nii.gz`` with the case's geometry."""
        prediction = self.predict(case_dir)
        template = sitk.ReadImage(os.path.join(str(case_dir), "Dose_Mask.nii.gz"))
        image = duplicate_image_metadata(template, sitk.GetImageFromArray(prediction))
        os.makedirs(str(out_dir), exist_ok=True)
        out_file = os.path.join(str(out_dir), "Dose.nii.gz")
        sitk.WriteImage(image, out_file)
        return out_file

    def predict_all(self, case_dirs, out_root):
        """Predict a whole list of cases, reusing the one loaded network."""
        written = []
        for case_dir in case_dirs:
            name = os.path.basename(str(case_dir).rstrip("/"))
            start = time.time()
            written.append(self.predict_to_nifti(case_dir, os.path.join(str(out_root), name)))
            print(f"    {name}  {time.time() - start:5.1f}s")
        return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cases",
        default="glioblastoma",
        help="directory under the data root holding the case folders",
    )
    ap.add_argument(
        "--weights",
        default=str(checkpoint_path("dose-predictor", "weights.pt")),
        help="dose-predictor weights (slim or original checkpoint)",
    )
    ap.add_argument(
        "--run",
        default="dose-predictor",
        help="name of the output run directory under the predictions root",
    )
    ap.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="case names; default is the 20 test cases DLDP_081..DLDP_100",
    )
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    ap.add_argument("--no-tta", action="store_true", help="disable test-time augmentation")
    ap.add_argument("--show-config", action="store_true", help="print resolved paths and exit")
    args = ap.parse_args()

    if args.show_config:
        print(describe())
        return

    subjects = args.subjects or ["DLDP_%03d" % i for i in range(81, 101)]
    case_dirs = [data_path(args.cases, s) for s in subjects]
    out_root = prediction_path(args.cases, args.run)

    predictor = DosePredictor(args.weights, device=args.device, tta=not args.no_tta)
    print(f"loaded {args.weights} on {predictor.device}")
    print(f"predicting {len(case_dirs)} cases -> {out_root}")
    predictor.predict_all(case_dirs, out_root)


if __name__ == "__main__":
    main()
