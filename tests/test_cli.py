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


def test_cli_generate_time_series_smoke(tmp_path):
    """Generate a tiny time-series dataset via CLI (T5 mode: mode: time_series)."""
    cfg = tmp_path / "tiny_ts.yaml"
    cfg.write_text(
        """
dataset_id: test-cli-ts
mode: time_series
instrument_config: configs/instruments/vi-da-easy-01.yaml
n_series: 2
n_scans_per_series: 5
scan_interval_s: 1.0
master_seed: 998
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

    from spektran.io import read_time_series

    records, interval = read_time_series(tmp_path / "out" / "test-cli-ts.h5")
    assert len(records) == 10
    assert interval == 1.0
    conc = [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
    assert conc[:5] == [conc[0]] * 5
    assert conc[5:] == [conc[5]] * 5
    assert conc[0] != conc[5]


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
