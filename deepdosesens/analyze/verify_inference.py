# -*- coding: utf-8 -*-
"""Check that the archived predictions can be regenerated from the archived weights.

Reproducing a table from archived predictions only shows the table matches the
predictions. This closes the loop: it loads each slimmed checkpoint, runs inference,
and compares the result with the prediction volume shipped in the archive.

    python -m deepdosesens.analyze.verify_inference
    python -m deepdosesens.analyze.verify_inference --cases 3 --model dose-predictor

Agreement is expected to a few times 1e-4 Gy on a 0-70 Gy scale: float32
arithmetic, and the accelerator, are not bit-reproducible across machines.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from deepdosesens.config import (  # noqa: E402
    CHECKPOINTS_DIR,
    data_path,
    describe,
    prediction_path,
    results_path,
)
from deepdosesens.inference import DosePredictor  # noqa: E402
from deepdosesens.visualization.panels import read  # noqa: E402

# Which archived prediction run each checkpoint produced. The ISBI paper's best run
# and the Cancers paper's initial model are the same weights, so that checkpoint is
# checked against both sets of archived volumes.
MODEL_PREDICTIONS = {
    "dose-predictor": ["run-6", "initial-model"],
    "concave-updated-model": ["concave-updated-model"],
    "multiple-lesion-updated-model": ["multiple-lesion-updated-model"],
    "combined-updated-model": ["combined-updated-model"],
}

TOLERANCE_GY = 1e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=2, help="cases per prediction run")
    ap.add_argument("--model", default="all", help="checkpoint name, or all")
    ap.add_argument("--device", default=None)
    ap.add_argument("--show-config", action="store_true")
    args = ap.parse_args()

    if args.show_config:
        print(describe())
        return

    wanted = MODEL_PREDICTIONS if args.model == "all" else {args.model: MODEL_PREDICTIONS[args.model]}
    rows = []
    for model, runs in wanted.items():
        weights = CHECKPOINTS_DIR / model / "weights.pt"
        if not weights.exists():
            print(f"{model}: no weights at {weights}; skipping")
            continue
        predictor = DosePredictor(weights, device=args.device)
        print(f"\n=== {model}  ({predictor.device})")

        for run in runs:
            run_dir = prediction_path("glioblastoma", run)
            if not run_dir.is_dir():
                print(f"    {run}: no archived predictions; skipping")
                continue
            cases = sorted(
                case.name
                for case in run_dir.iterdir()
                if case.is_dir() and data_path("glioblastoma", case.name).is_dir()
            )[: args.cases]
            for case in cases:
                start = time.time()
                predicted = predictor.predict(data_path("glioblastoma", case))
                elapsed = time.time() - start
                archived = read(run_dir / case / "Dose.nii.gz")
                body = read(data_path("glioblastoma", case, "Dose_Mask.nii.gz")) > 0
                reference = read(data_path("glioblastoma", case, "Dose.nii.gz"))
                rows.append(
                    {
                        "model": model,
                        "archived run": run,
                        "case": case,
                        "max |diff| (Gy)": float(np.abs(predicted - archived).max()),
                        "dose score, recomputed": float(
                            np.abs(predicted - reference)[body].mean()
                        ),
                        "dose score, archived": float(
                            np.abs(archived - reference)[body].mean()
                        ),
                        "seconds": elapsed,
                    }
                )
                row = rows[-1]
                print(
                    f"    {run:<30} {case}"
                    f"  max|diff| {row['max |diff| (Gy)']:.2e} Gy"
                    f"   dose score {row['dose score, recomputed']:.4f}"
                    f" vs archived {row['dose score, archived']:.4f}"
                    f"   ({elapsed:.0f}s)"
                )

    report = pd.DataFrame(rows)
    if report.empty:
        print("\nnothing to verify: no weights found; run scripts/fetch_artifacts.sh.")
        return
    report["verdict"] = np.where(
        report["max |diff| (Gy)"] <= TOLERANCE_GY, "matches", "DIFFERS"
    )
    os.makedirs(results_path(), exist_ok=True)
    report.to_csv(results_path("inference_verification.csv"), index=False)

    print("\n" + "=" * 78)
    matched = (report["verdict"] == "matches").sum()
    print(f"{matched} of {len(report)} regenerated volumes match the archive")
    print(f"worst disagreement {report['max |diff| (Gy)'].max():.2e} Gy on a 0-70 Gy scale")
    print("=" * 78)
    if matched != len(report):
        print(report[report["verdict"] == "DIFFERS"].to_string(index=False))
    print(f"\nwritten to {results_path('inference_verification.csv')}")


if __name__ == "__main__":
    main()
