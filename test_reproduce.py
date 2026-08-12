"""Smoke-test the reproduction and video scripts.

Large artifacts are not committed, so a script whose inputs are absent is skipped
rather than failed. Fetch the artifacts first to exercise them for real:

    scripts/fetch_artifacts.sh
"""

import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from deepdosesens.config import (  # noqa: E402
    CHECKPOINTS_DIR,
    data_path,
    prediction_path,
)

# Every runnable module, with one representative input it cannot work without.
# Adding a module without declaring its inputs is caught by test_all_declared.
MODULES = {
    "deepdosesens.config": None,
    "deepdosesens.analyze.reproduce_isbi_tables": prediction_path("glioblastoma", "run-6"),
    "deepdosesens.analyze.reproduce_cancers_tables": prediction_path(
        "glioblastoma", "initial-model"
    ),
    "deepdosesens.analyze.verify_inference": CHECKPOINTS_DIR / "dose-predictor",
    "deepdosesens.inference": CHECKPOINTS_DIR / "dose-predictor",
    "deepdosesens.visualization.make_dose_video": data_path("glioblastoma"),
    "deepdosesens.visualization.make_sensitivity_video": data_path("optic-nerve-variants"),
    "deepdosesens.visualization.make_robustness_video": prediction_path(
        "glioblastoma", "concave-updated-model"
    ),
}


def run(args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", *args], capture_output=True, text=True, env=env, cwd=REPO_ROOT
    )


class TestScriptsImportAndConfigure(unittest.TestCase):
    def test_all_declared(self):
        """Every module under analyze/ and visualization/ must be declared above."""
        declared = set(MODULES)
        found = set()
        for package in ("analyze", "visualization"):
            directory = os.path.join(REPO_ROOT, "deepdosesens", package)
            for name in os.listdir(directory):
                if not name.endswith(".py") or name.startswith("_"):
                    continue
                stem = name[:-3]
                # Library modules have no main; only entry points need declaring.
                with open(os.path.join(directory, name), encoding="utf-8") as handle:
                    if '__name__ == "__main__"' not in handle.read():
                        continue
                found.add(f"deepdosesens.{package}.{stem}")
        self.assertFalse(
            found - declared,
            f"add these to MODULES so the test knows what they need: "
            f"{sorted(found - declared)}",
        )

    def test_show_config(self):
        """Each entry point resolves its paths without touching the data."""
        for module in MODULES:
            if module == "deepdosesens.config":
                continue
            with self.subTest(module=module):
                result = run([module, "--show-config"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("data ", result.stdout)

    def test_config_describe(self):
        result = run(["deepdosesens.config"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for label in ("repo", "data", "checkpoints", "predictions", "results"):
            self.assertIn(label, result.stdout)


class TestReproduction(unittest.TestCase):
    """Run the reproduction scripts where their inputs are present."""

    def _run_if_available(self, module, extra=()):
        needed = MODULES[module]
        if needed is None or not os.path.isdir(needed):
            self.skipTest(
                f"{module} needs {needed}, which is not committed; "
                "fetch it with scripts/fetch_artifacts.sh"
            )
        result = run([module, *extra])
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        return result.stdout

    def test_isbi_tables(self):
        output = self._run_if_available("deepdosesens.analyze.reproduce_isbi_tables")
        # A few reported values differ from the archived artifacts; anything beyond
        # that count is a regression in the scoring code.
        self.assertIn("of 68 reported values reproduce", output)
        reproduced = int(output.split("of 68 reported values")[0].strip().split("\n")[-1])
        self.assertGreaterEqual(reproduced, 64, output[-3000:])

    def test_cancers_tables(self):
        output = self._run_if_available("deepdosesens.analyze.reproduce_cancers_tables")
        self.assertIn("of 88 reported values reproduce", output)
        reproduced = int(output.split("of 88 reported values")[0].strip().split("\n")[-1])
        self.assertGreaterEqual(reproduced, 64, output[-3000:])

    def test_scores_conventions(self):
        """The score definitions must match the archive's own per-case score files."""
        import pandas as pd

        from deepdosesens.analyze.scores import case_scores

        run_dir = prediction_path("glioblastoma", "initial-model")
        if not (run_dir / "dose_score.csv").exists():
            self.skipTest("archived per-case score files not present")
        case = "DLDP_081"
        dose, dvh = case_scores(data_path("glioblastoma", case), run_dir / case)
        archived_dose = pd.read_csv(run_dir / "dose_score.csv", index_col=0)
        archived_dvh = pd.read_csv(run_dir / "dvh_score.csv", index_col=0)
        self.assertAlmostEqual(dose["Body"], archived_dose.loc[case, "overall"], places=4)
        self.assertAlmostEqual(dvh["Overall"], archived_dvh.loc[case, "overall"], places=4)


class TestVideoRendering(unittest.TestCase):
    def test_single_frame_renders(self):
        """One frame of the dose sweep, to catch drawing errors without a full clip."""
        case = "DLDP_081"
        if not data_path("glioblastoma", case).is_dir():
            self.skipTest("planning cases not present")
        if not prediction_path("glioblastoma", "run-6", case).is_dir():
            self.skipTest("predictions not present")

        import tempfile

        from deepdosesens.visualization.make_dose_video import draw_frame, load_case

        data = load_case(
            data_path("glioblastoma", case), prediction_path("glioblastoma", "run-6", case)
        )
        with tempfile.TemporaryDirectory() as scratch:
            out = os.path.join(scratch, "frame.png")
            draw_frame(case, data, data["slices"][len(data["slices"]) // 2], out, 20.0, 65.0)
            self.assertGreater(os.path.getsize(out), 10_000)


if __name__ == "__main__":
    unittest.main()
