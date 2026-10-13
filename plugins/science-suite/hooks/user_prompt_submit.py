#!/usr/bin/env python3
"""UserPromptSubmit hook for science-suite: a conditional routing hint.

A skill-comply run (an external plugin, not in this repo) found models skip the hub routing tree
and start coding. A standing reminder fixed that but cost ~120 tokens on every prompt. This hook
injects one short line only when the prompt matches a registered hub's domain, names just the
matching hubs, and repeats a hub's hint only after REPEAT_AFTER prompts. Prompts that start with
"/" (explicit commands) are left alone.
"""

SUITE = "science-suite"
FALLBACK = "science-hub"
HUBS = {
    "jax-computing": r"\bjax\b|\bjit\b|vmap|pmap|flax|optax|diffrax|equinox",
    "julia-language": r"\bjulia\b|\.jl\b|multiple dispatch|\bpkg\.|project\.toml|documenter|pkgtemplates|precompil",
    "sciml-and-diffeq": r"differentialequations|diffeq|modelingtoolkit|sciml|universal differential|\bode\b|\bpde\b|neuralpde|\bsindy\b",
    "bayesian-inference": r"bayes|\bmcmc\b|\bnuts\b|posterior|turing\.jl|numpyro|variational inference|pigeons|hierarchical model",
    "deep-learning-hub": r"neural net|transformer|\bcnn\b|\bgnn\b|backprop|attention|training loop|diffusion model|pytorch|\blstm\b|autoencoder",
    "ml-and-data-science": r"scikit|sklearn|xgboost|lightgbm|random forest|cross-valid|feature engineering|hyperparameter|mlflow|pandas|dataframe|curve fit|\bzarr\b|hdf5|netcdf|\.h5\b|\.hdf\b|\.nc\b|chunking",
    "ml-deployment": r"torchserve|model serving|\bonnx\b|quantiz|mlops|triton",
    "nonlinear-dynamics": r"bifurcation|chaos|lyapunov|attractor|synchroni[sz]|oscillator|pattern formation|dynamical system",
    "statistical-physics-hub": r"statistical (mechanics|physics)|partition function|phase transition|langevin|fokker|fluctuation|correlation function|active matter|non-?equilibrium|brownian|soft matter|\bxpcs\b|\bdls\b|\bsaxs\b",
    "simulation-and-hpc": r"molecular dynamics|\bmd\b|gromacs|openmm|lammps|force field|trajectory|monte carlo|\bhpc\b|slurm|\bmpi\b|\bcuda\b",
    "continuum-mechanics-and-rheology": r"finite element|\bfem\b|\bfea\b|constitutive|viscoelastic|rheolog|\bdma\b|hyperelastic|vitrimer|nanocomposite",
    "llm-and-ai": r"\bllms?\b|\brag\b|langchain|embedding|retrieval|\bnlp\b",
    "time-series-analysis": r"time series|arima|garch|forecast|stationarity|prophet",
    "python-development": r"pyo3|rust extension|type hints|\bmypy\b|\bruff\b|pyproject|taskgroup",
    "parallel-computing": r"parallel|openmp|multithread|distributed",
}

import json
import os
import re
import sys
import tempfile
from pathlib import Path

from _hook_io import get_field, read_payload, wrap_context

REPEAT_AFTER = 12  # prompts before the same hub is hinted again
MAX_HUBS = 3

PATTERNS = {hub: re.compile(rx, re.IGNORECASE) for hub, rx in HUBS.items()}


def classify(prompt: str) -> list[str]:
    """Hubs whose domain the prompt touches, best match first."""
    scores = {hub: len(rx.findall(prompt)) for hub, rx in PATTERNS.items()}
    return [h for h, n in sorted(scores.items(), key=lambda kv: -kv[1]) if n][:MAX_HUBS]


def due_hubs(matched: list[str], session_id: str) -> list[str]:
    """Matched hubs not hinted in the last REPEAT_AFTER prompts of this session (state lives in the temp dir)."""
    uid = getattr(os, "getuid", lambda: 0)()
    base = Path(os.environ.get("MYCLAUDE_HINT_DIR") or tempfile.gettempdir()) / f"myclaude-route-hints-{uid}"
    try:
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        return matched  # no writable state: hint every time rather than fail the prompt
    path = base / f"{SUITE}-{re.sub(r'[^A-Za-z0-9_.-]', '_', session_id)[:80]}.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {}
    n = int(state.get("n", 0)) + 1
    last = state.get("last", {})
    due = [h for h in matched if n - int(last.get(h, -REPEAT_AFTER)) >= REPEAT_AFTER]
    last.update({h: n for h in due})
    try:
        path.write_text(json.dumps({"n": n, "last": last}))
    except OSError:
        pass  # state is a convenience; a failed write only means the hint may repeat
    return due


def main() -> None:
    try:
        payload = read_payload()
        prompt = payload.get("prompt")
        result: dict = {"status": "success"}
        if isinstance(prompt, str) and prompt.strip() and not prompt.lstrip().startswith("/"):
            hubs = due_hubs(classify(prompt), get_field(payload, "session_id", default="unknown"))
            if hubs:
                hint = (
                    f"Routing hint: this touches {', '.join(f'{SUITE}:{h}' for h in hubs)}. "
                    f"Load the matching hub skill before implementing ({SUITE}:{FALLBACK} if unsure)."
                )
                result["additionalContext"] = hint
                result.update(wrap_context("UserPromptSubmit", hint))
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"UserPromptSubmit hook error: {e}", file=sys.stderr)
        json.dump({"status": "error", "message": f"UserPromptSubmit hook error: {e}"}, sys.stdout)


if __name__ == "__main__":
    main()
