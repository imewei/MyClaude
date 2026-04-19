#!/usr/bin/env python3
"""
observable_extractor.py

Scaffold for extracting the predicted observable from a simulation trajectory
in the format Stage 7 (experiment-designer) expects. The actual extraction
logic is problem-specific; this module provides the structure and the output
formatting so that every Stage 6 prototype emits a predicted observable in
the same schema, regardless of what the observable actually is.

The typical integration is:

    from observable_extractor import ObservableBuilder, write_predicted_observable
    from mymod.core import integrate, extract_my_observable

    # Run the prototype
    trajectory = integrate(...)

    # Your problem-specific extraction
    t_array, values_array = extract_my_observable(trajectory)

    # Build the standardized observable record
    obs = (
        ObservableBuilder("spectral_gap_time_series")
        .set_type("time_series")
        .set_physical_meaning(
            "First-to-second eigenvalue gap of the stress-response operator, "
            "predicted to collapse >30s before flocculation onset."
        )
        .set_time_series(t_array, values_array, t_units="seconds", value_units="dimensionless")
        .set_uncertainty("numerical", "relative", 0.03, "convergence study dt=1e-4 to 1e-2")
        .set_uncertainty("parametric", "relative", 0.12, "phi range 0.45-0.55")
        .set_uncertainty("statistical", "absolute", 0.02, "ensemble of 200 realizations")
        .set_temporal_structure(characteristic_timescale_s=2.0, total_duration_s=120.0)
        .set_noise_model("speckle", fully_developed=True)
        .set_expected_snr(15)
        .build()
    )

    write_predicted_observable(obs, "artifacts/predicted_observable.yaml")
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class ObservableBuilder:
    """Fluent builder that produces a predicted-observable record matching
    templates/predicted_observable.md."""

    name: str
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        self.data = {"name": self.name, "uncertainty": {}}

    def set_type(self, type_: str) -> "ObservableBuilder":
        allowed = {"time_series", "spectrum", "scalar", "tensor", "distribution"}
        if type_ not in allowed:
            raise ValueError(f"type_ must be one of {sorted(allowed)}, got {type_!r}")
        self.data["type"] = type_
        return self

    def set_physical_meaning(self, meaning: str) -> "ObservableBuilder":
        self.data["physical_meaning"] = meaning
        return self

    def set_time_series(
        self,
        t,
        values,
        t_units: str = "seconds",
        value_units: str = "",
    ) -> "ObservableBuilder":
        self.data["predicted_values"] = {
            "t": _to_list(t),
            "values": _to_list(values),
            "units": {"t": t_units, "values": value_units},
        }
        return self

    def set_scalar(self, value: float, units: str = "") -> "ObservableBuilder":
        self.data["predicted_values"] = {
            "value": float(value),
            "units": units,
        }
        return self

    def set_spectrum(
        self,
        freq,
        amplitude,
        freq_units: str = "Hz",
        amp_units: str = "",
    ) -> "ObservableBuilder":
        self.data["predicted_values"] = {
            "freq": _to_list(freq),
            "amplitude": _to_list(amplitude),
            "units": {"freq": freq_units, "amplitude": amp_units},
        }
        return self

    def set_uncertainty(
        self,
        source: str,
        type_: str,
        value: float,
        origin: str,
    ) -> "ObservableBuilder":
        """source: 'numerical', 'parametric', or 'statistical'.
        type_: 'relative' or 'absolute'."""
        allowed_sources = {"numerical", "parametric", "statistical"}
        if source not in allowed_sources:
            raise ValueError(
                f"source must be one of {sorted(allowed_sources)}, got {source!r}"
            )
        allowed_types = {"relative", "absolute"}
        if type_ not in allowed_types:
            raise ValueError(
                f"type_ must be one of {sorted(allowed_types)}, got {type_!r}"
            )
        self.data["uncertainty"][source] = {
            "type": type_,
            "value": float(value),
            "source": origin,
        }
        return self

    def set_temporal_structure(
        self,
        characteristic_timescale_s: float | None = None,
        min_sampling_rate_hz: float | None = None,
        total_duration_s: float | None = None,
    ) -> "ObservableBuilder":
        if characteristic_timescale_s is not None and min_sampling_rate_hz is None:
            min_sampling_rate_hz = 10.0 / characteristic_timescale_s
        entry = {}
        if characteristic_timescale_s is not None:
            entry["characteristic_timescale_s"] = float(characteristic_timescale_s)
        if min_sampling_rate_hz is not None:
            entry["min_sampling_rate_hz"] = float(min_sampling_rate_hz)
        if total_duration_s is not None:
            entry["total_duration_s"] = float(total_duration_s)
        self.data["temporal_structure"] = entry
        return self

    def set_spatial_structure(
        self,
        characteristic_length_m: float | None = None,
        min_resolution_m: float | None = None,
        extent_m: float | None = None,
    ) -> "ObservableBuilder":
        entry = {}
        if characteristic_length_m is not None:
            entry["characteristic_length_m"] = float(characteristic_length_m)
            if min_resolution_m is None:
                entry["min_resolution_m"] = float(characteristic_length_m / 10.0)
        if min_resolution_m is not None:
            entry["min_resolution_m"] = float(min_resolution_m)
        if extent_m is not None:
            entry["extent_m"] = float(extent_m)
        self.data["spatial_structure"] = entry
        return self

    def set_noise_model(self, type_: str, **kwargs) -> "ObservableBuilder":
        self.data["noise_model"] = {"type": type_, **kwargs}
        return self

    def set_expected_snr(self, snr: float) -> "ObservableBuilder":
        self.data["expected_snr_at_typical_conditions"] = float(snr)
        return self

    def build(self) -> dict:
        # Validate required fields
        required = ["name", "type", "physical_meaning", "predicted_values", "uncertainty"]
        missing = [f for f in required if f not in self.data]
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        if not self.data["uncertainty"]:
            raise ValueError(
                "uncertainty dict is empty; set at least one of "
                "numerical / parametric / statistical"
            )
        return dict(self.data)


def _to_list(x) -> list:
    """Best-effort conversion to a plain Python list for YAML/JSON serialization."""
    try:
        return x.tolist()  # numpy or jax array
    except AttributeError:
        return list(x)


def write_predicted_observable(obs: dict, path: Path | str) -> None:
    """Write the observable record to YAML (preferred) or JSON based on extension."""
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        if not _HAS_YAML:
            raise RuntimeError("pyyaml required to write YAML; use .json extension instead")
        path.write_text(yaml.safe_dump(obs, sort_keys=False), encoding="utf-8")
    elif path.suffix == ".json":
        path.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"unsupported extension {path.suffix}; use .yaml or .json")


def load_predicted_observable(path: Path | str) -> dict:
    """Load a predicted observable record from YAML or JSON."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        if not _HAS_YAML:
            raise RuntimeError("pyyaml required to read YAML")
        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"unsupported extension {path.suffix}")


# Lightweight self-check runnable
def _demo() -> int:
    obs = (
        ObservableBuilder("demo_observable")
        .set_type("scalar")
        .set_physical_meaning("Demo observable for self-test of the builder.")
        .set_scalar(1.234, units="dimensionless")
        .set_uncertainty("numerical", "relative", 0.01, "demo")
        .set_expected_snr(10.0)
        .build()
    )
    print(json.dumps(obs, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_demo())
