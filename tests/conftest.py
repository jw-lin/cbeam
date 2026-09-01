"""Shared pytest fixtures and configuration for the cbeam test-suite.

The suite is split in two:

* ``tests/unit``        - fast, isolated tests of individual classes / functions.
* ``tests/integration`` - end-to-end tests, each one a rewrite of an example
                          from ``docs_source`` (the Sphinx documentation).

Anything that runs a full ``Propagator.characterize`` / ``compute_modes`` on a
tapering waveguide is marked ``slow``.  Run only the quick tests with::

    pytest -m "not slow"

and the whole thing (minutes, not seconds) with a plain ``pytest``.
"""

from __future__ import annotations

import os
import sys
import pathlib

import pytest

# --------------------------------------------------------------------------- #
# Head-less plotting.  cbeam calls into matplotlib in a lot of places; force a
# non-interactive backend before anything imports pyplot.
# --------------------------------------------------------------------------- #
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------------- #
# The FEval submodule activates a small Julia project living next to the
# package (``src/cbeam/FEval``).  A fresh checkout only ships ``Project.toml``;
# the project has to be instantiated once so that a ``Manifest.toml`` exists.
# Do it here so ``import cbeam.FEval`` works out of the box on CI / a clean box.
# --------------------------------------------------------------------------- #
def _ensure_feval_instantiated() -> None:
    feval_dir = SRC / "cbeam" / "FEval"
    if (feval_dir / "Manifest.toml").exists():
        return
    try:
        from juliacall import Main as jl
    except Exception as exc:  # pragma: no cover - juliacall missing entirely
        pytest.skip(f"juliacall unavailable, cannot set up FEval: {exc}",
                    allow_module_level=True)
        return
    jl.seval("using Pkg")
    jl.Pkg.activate(str(feval_dir))
    jl.Pkg.instantiate()
    jl.Pkg.precompile()


def pytest_configure(config: pytest.Config) -> None:
    _ensure_feval_instantiated()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def save_dir(tmp_path) -> str:
    """A throw-away ``save_dir`` for a ``Propagator`` so nothing is written to
    the repo (the default is ``./data``)."""
    d = tmp_path / "data"
    return str(d)


@pytest.fixture(autouse=True)
def _close_figures():
    """Never leak matplotlib figures between tests."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


@pytest.fixture(scope="session")
def np():
    import numpy as _np

    return _np
