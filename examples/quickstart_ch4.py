"""Phase 0 acceptance demo: 10 lines to a schema-valid CH4 absorption record."""

import uuid

from spektran.physics import simulate_absorbance
from spektran.validate import validate_record

nu, absorbance = simulate_absorbance(
    molecule="CH4", concentration_ppm=100.0, temperature_K=296.0,
    pressure_atm=1.0, path_length_m=10.0,
    wavenumber_start_cm1=6046.0, wavenumber_end_cm1=6048.0,
)

record = {
    "record_id": str(uuid.uuid4()),
    "schema_version": "0.1",
    "data_origin": "simulated",
    "technique": "TDLAS-DA",
    "provenance": {
        "generator_version": "0.1.0.dev0",
        "hitran_data_version": "n/a (built-in demo lines)",
        "random_seed": 0,
        "instrument_config_id": "vi-clean-00",
        "noise_config": {},
    },
    "signals": {
        "absorbance": {
            "array_ref": "/records/demo/absorbance",
            "n_samples": int(len(nu)),
            "wavenumber_axis": {
                "start_cm1": float(nu[0]),
                "step_cm1": float(nu[1] - nu[0]),
            },
        }
    },
    "labels": {
        "species": [
            {"molecule": "CH4", "hitran_molecule_id": 6, "concentration_ppm": 100.0}
        ]
    },
    "conditions": {
        "temperature_K": 296.0,
        "pressure_atm": 1.0,
        "path_length_m": 10.0,
        "matrix_gas": "N2",
    },
    "instrument": {
        "laser": {"center_wavenumber_cm1": 6047.0, "scan_range_cm1": 2.0},
        "detector": {"type": "InGaAs photodiode"},
        "target_lines": [{"hitran_molecule_id": 6, "wavenumber_cm1": 6046.9647}],
    },
}

errors = validate_record(record)
print(f"points: {len(nu)}, peak absorbance: {absorbance.max():.4e}")
print("schema validation:", "PASS" if not errors else errors)
