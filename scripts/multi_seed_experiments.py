#!/usr/bin/env python
"""Run multi-seed experiments for DL baselines to get error bars.

Runs each DL baseline (Transformer T1, CNN T1, U-Net T2, TCN T5) with
3 different random seeds and reports mean +/- std for the primary metric.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 2024, 7777]

EXPERIMENTS = [
    {
        "name": "Transformer T1",
        "script": "baselines/transformer_t1/train.py",
        "seed_var": "SEED",
        "metric_file": "baselines/transformer_t1/predictions_t1-test.csv",
        "truth": "data/ch4-t1-test-v0.h5",
        "task": "T1-concentration",
    },
    {
        "name": "CNN T1",
        "script": "baselines/cnn1d/train.py",
        "seed_var": "SEED",
        "metric_file": "baselines/cnn1d/predictions_t1-test.csv",
        "truth": "data/ch4-t1-test-v0.h5",
        "task": "T1-concentration",
    },
]


def run_with_seed(script: str, seed: int, seed_var: str) -> dict | None:
    """Run a training script with a modified seed and return scores."""
    script_path = REPO / script
    content = script_path.read_text()

    original_seed_line = None
    for line in content.split("\n"):
        if line.strip().startswith(f"{seed_var} ="):
            original_seed_line = line
            break

    if original_seed_line is None:
        print(f"  WARNING: Could not find {seed_var} in {script}")
        return None

    modified = content.replace(original_seed_line, f"{seed_var} = {seed}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(modified)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode != 0:
            print(f"  FAILED (seed={seed}): {result.stderr[-200:]}")
            return None

        out_dir = REPO / Path(script).parent
        for score_file in out_dir.glob("scores_*.json"):
            scores = json.loads(score_file.read_text())
            return scores

        print(f"  No scores file found for seed={seed}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (seed={seed})")
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> int:
    results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")

        seed_results = []
        for seed in SEEDS:
            print(f"  Running seed={seed}...")
            scores = run_with_seed(exp["script"], seed, exp["seed_var"])
            if scores:
                seed_results.append(scores)
                mae = scores.get("mae_ppm", scores.get("test_mae_ppm", "?"))
                print(f"  seed={seed}: MAE={mae}")

        if seed_results:
            maes = [s.get("mae_ppm", s.get("test_mae_ppm", 0)) for s in seed_results]
            results[exp["name"]] = {
                "maes": maes,
                "mean": float(np.mean(maes)),
                "std": float(np.std(maes)),
                "n_seeds": len(maes),
            }
            print(f"\n  {exp['name']}: MAE = {np.mean(maes):.2f} +/- {np.std(maes):.2f} ppm")

    out_path = REPO / "baselines" / "multi_seed_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path.relative_to(REPO)}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: {r['mean']:.2f} +/- {r['std']:.2f} ppm (n={r['n_seeds']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
