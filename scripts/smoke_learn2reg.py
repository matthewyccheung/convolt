from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> None:
    py = sys.executable

    datasets = ["hippocampusmr", "oasis", "abdomenctct"]
    for d in datasets:
        results_dir = Path("/tmp") / f"smoke_{d}_demons_atlas-single"
        uq_dir = Path("/tmp") / f"smoke_{d}_demons_atlas-single_uq"

        _run(
            [
                py,
                "-m",
                "reg",
                "register",
                "--dataset",
                d,
                "--method",
                "demons",
                "--split",
                "training",
                "--atlas_mode",
                "single",
                "--atlas_n",
                "1",
                "--max_pairs",
                "4",
                "--results_dir",
                str(results_dir),
                "--no_nifti_outputs",
            ]
        )

        _assert_exists(results_dir / "summary.csv")
        _assert_exists(results_dir / "label_volumes.csv")
        _assert_exists(results_dir / "atlas_meta.json")
        # At least one pair artifact + figure.
        _assert_exists(next((results_dir / "pairs").glob("*/artifacts.npz")))
        _assert_exists(next((results_dir / "figures").glob("*.png")))

        _run(
            [
                py,
                "-m",
                "reg.uq.cli",
                "--results_dir",
                str(results_dir),
                "--out_dir",
                str(uq_dir),
                "--uq_target",
                "volume_union",
                "--alpha",
                "0.1",
                "--n_repeats",
                "1",
                "--n_train",
                "0",
                "--n_calib",
                "2",
                "--n_test",
                "2",
                "--min_calib",
                "1",
                "--min_test",
                "1",
                "--beta_model",
                "none",
            ]
        )

        _assert_exists(uq_dir / "cp_summary.csv")
        _assert_exists(uq_dir / "cp_runs.csv")
        _assert_exists(uq_dir / "method_table.csv")
        _assert_exists(uq_dir / "intervals_test.csv")

    print("Smoke check OK", flush=True)


if __name__ == "__main__":
    main()
