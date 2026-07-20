from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from granite_ml.config import LOG_DIR, RAW_DATA_FILE, ensure_project_directories


STEPS = {
    "00": "00_data_audit.py",
    "01": "01_preprocess.py",
    "02": "02_normality.py",
    "03": "03_correlations.py",
    "04": "04_select_cluster_champions.py",
    "05": "05_summarize_stability.py",
    "06": "06_feature_contribution.py",
    "07": "07_exploratory_ratio_statistics.py",
    "08": "08_model_comparison.py",
    "09": "09_class_weight_sensitivity.py",
    "10": "10_traditional_baseline.py",
    "11": "11_final_shap.py",
    "12": "12_generate_summary_figures.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the granite A/S/I classification pipeline.")
    parser.add_argument("--steps", nargs="+", choices=STEPS, help="Run only the listed step numbers.")
    parser.add_argument("--from-step", choices=STEPS, help="Run this step and every later step.")
    return parser.parse_args()


def selected_steps(args: argparse.Namespace) -> list[str]:
    ordered = list(STEPS)
    if args.steps:
        requested = set(args.steps)
        return [step for step in ordered if step in requested]
    if args.from_step:
        return ordered[ordered.index(args.from_step):]
    return ordered


def run_step(step: str) -> None:
    script = PROJECT_ROOT / "scripts" / STEPS[step]
    log_path = LOG_DIR / f"step_{step}.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONIOENCODING"] = "utf-8"
    print(f"\n{'=' * 72}\nRunning step {step}: {script.name}\nLog: {log_path}\n{'=' * 72}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"Step {step} failed with exit code {return_code}. See {log_path}")


def main() -> None:
    ensure_project_directories()
    args = parse_args()
    steps = selected_steps(args)
    if any(step in {"00", "01"} for step in steps) and not RAW_DATA_FILE.exists():
        raise SystemExit(
            "Missing analysis-ready dataset. Copy it to:\n"
            f"  {RAW_DATA_FILE}\n"
            "Expected filename: SCB-Mesozoic-Granite.csv"
        )
    for step in steps:
        run_step(step)
    print("\nAll selected steps completed successfully.")


if __name__ == "__main__":
    main()
