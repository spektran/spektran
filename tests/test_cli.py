"""CLI integration tests."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_cli_help_shows_generate():
    result = subprocess.run(
        [sys.executable, "-m", "spektran.cli", "--help"],
        capture_output=True, text=True,
    )
    assert "generate" in result.stdout


def test_cli_generate_smoke(tmp_path):
    """Generate a tiny dataset via CLI."""
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(
        """
dataset_id: test-cli
instrument_config: configs/instruments/vi-da-easy-01.yaml
n_records: 5
master_seed: 999
gas:
  molecule: CH4
  concentration_ppm: {low: 50.0, high: 150.0, log_uniform: false}
  path_length_m: 5.0
  matrix_gas: N2
n_points: 200
line_source: demo
"""
    )
    result = subprocess.run(
        [sys.executable, "-m", "spektran.cli", "generate", str(cfg),
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (tmp_path / "out" / "test-cli.h5").exists()


def test_cli_benchmark_run_help():
    result = subprocess.run(
        [sys.executable, "-m", "spektran.cli", "benchmark", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "spektran.cli", "--version"],
        capture_output=True, text=True,
    )
    from spektran import __version__
    assert __version__ in result.stdout
