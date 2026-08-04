"""Virtual-instrument parameter sampling: distributions -> concrete values.

A virtual instrument (YAML validated by instrument.schema.json) specifies a
DISTRIBUTION for every parameter; each record samples one concrete value from
each distribution and stores all sampled values in provenance.noise_config
(plan §5.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def sample_value(spec, rng: np.random.Generator):
    """Sample one concrete value from a dist_or_number spec."""
    if isinstance(spec, (int, float)):
        return float(spec)
    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError(f"Invalid distribution spec: {spec!r}")
    d = spec["dist"]
    if d == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if d == "normal":
        return float(rng.normal(spec["mean"], spec["sigma"]))
    if d == "loguniform":
        return float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
    if d == "choice":
        return float(rng.choice(spec["values"]))
    raise ValueError(f"Unknown distribution: {d!r}")


def sample_tree(node, rng: np.random.Generator):
    """Recursively sample every dist_or_number leaf of a config subtree."""
    if isinstance(node, dict):
        if "dist" in node:
            return sample_value(node, rng)
        return {k: sample_tree(v, rng) for k, v in node.items()}
    if isinstance(node, list):
        return [sample_tree(v, rng) for v in node]
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return float(node)
    return node


def load_instrument_config(path: str | Path) -> dict:
    """Load and schema-validate a virtual-instrument YAML config."""
    from ..validate import instrument_validator

    cfg = yaml.safe_load(Path(path).read_text())
    errors = list(instrument_validator().iter_errors(cfg))
    if errors:
        msgs = "; ".join(e.message for e in errors[:5])
        raise ValueError(f"Instrument config {path} invalid: {msgs}")
    return cfg


def sample_instrument(cfg: dict, rng: np.random.Generator) -> dict:
    """Sample a concrete instrument realization from a validated config.

    Returns a plain dict mirroring the config structure with every
    distribution replaced by a sampled number. Non-parameter metadata keys
    are passed through untouched.
    """
    meta_keys = {"instrument_config_id", "schema_version", "technique", "held_out",
                 "description", "performance"}
    out = {k: cfg[k] for k in meta_keys if k in cfg}
    for key, sub in cfg.items():
        if key not in meta_keys:
            out[key] = sample_tree(sub, rng)
    return out
