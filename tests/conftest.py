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
import re
import sys
import pathlib

import numpy as np_
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


# --------------------------------------------------------------------------- #
# Numerical regression ("golden file") helper.
#
# The integration tests assert on physical invariants (power conservation, index
# ordering, ...) with loose tolerances so they survive a mesh/eigenbasis change.
# The ``golden`` fixture adds a second, stricter layer: it records the actual
# numbers a run produces and, on later runs, checks they have not drifted.
#
#     CBEAM_GOLDEN=record   compute + write tests/integration/_golden/<id>.npz
#     CBEAM_GOLDEN=check     (default) load + np.testing.assert_allclose;
#                            a missing golden file skips the assertion
#     CBEAM_GOLDEN=off       no-op (the plain invariant asserts still run)
#
# The reference set under _golden/ is generated from the ``tests`` branch and
# committed, so a run on any other branch is compared against it.
# --------------------------------------------------------------------------- #
GOLDEN_DIR = REPO_ROOT / "tests" / "integration" / "_golden"
GOLDEN_MODE = os.environ.get("CBEAM_GOLDEN", "check").lower()


class _Golden:
    """Per-test recorder / comparator handed to a test by the ``golden`` fixture."""

    def __init__(self, path: pathlib.Path, mode: str):
        self._path = path
        self._mode = mode
        self._new: dict = {}          # record mode: name -> array
        self._ref: dict | None = None  # check mode: lazily loaded npz

    # -- internal ---------------------------------------------------------
    def _load_ref(self) -> dict:
        if self._ref is None:
            if not self._path.exists():
                pytest.skip(
                    f"no golden file {self._path.name}; regenerate the reference "
                    f"set with CBEAM_GOLDEN=record"
                )
            self._ref = dict(np_.load(self._path, allow_pickle=False))
        return self._ref

    def _prep(self, value, abs_compare: bool, sort: bool):
        arr = np_.asarray(value, dtype=float)
        if abs_compare:
            arr = np_.abs(arr)
        if sort:
            arr = np_.sort(arr, axis=None) if arr.ndim <= 1 else np_.sort(arr, axis=-1)
        return arr

    # -- public API -----------------------------------------------------
    def check(self, name, value, *, rtol=1e-6, atol=1e-9,
              abs_compare=False, sort=False):
        """Record ``value`` under ``name`` (record mode) or assert it matches the
        stored golden array (check mode).

        abs_compare : compare magnitudes  (eigenvector / mode-amplitude sign is
                      arbitrary).
        sort        : compare the sorted values (mode / channel order within a
                      symmetric device is arbitrary).
        """
        arr = self._prep(value, abs_compare, sort)
        if self._mode == "off":
            return
        if self._mode == "record":
            self._new[name] = arr
            return
        ref = self._load_ref()
        assert name in ref, f"golden {self._path.name} is missing key {name!r}"
        exp = ref[name]
        assert exp.shape == arr.shape, (
            f"{name}: value shape {arr.shape} != golden shape {exp.shape}"
        )
        np_.testing.assert_allclose(
            arr, exp, rtol=rtol, atol=atol,
            err_msg=f"{name}: max|delta|={np_.abs(arr - exp).max():.3e}",
        )

    def check_vs_z(self, name, zs, arr, *, rtol=1e-4, atol=1e-6,
                   abs_compare=False):
        """Like :meth:`check` for a z-series ``arr`` of shape ``(len(zs), M)``.

        The golden array is linearly resampled onto the current ``zs`` over the
        overlapping z-range first, so a changed adaptive step schedule does not
        by itself trip the comparison.
        """
        zs = np_.asarray(zs, dtype=float)
        arr = self._prep(arr, abs_compare, sort=False)
        if self._mode == "off":
            return
        if self._mode == "record":
            self._new[name] = arr
            self._new[name + "__z"] = zs
            return
        ref = self._load_ref()
        exp, exp_z = ref[name], ref[name + "__z"]
        lo, hi = max(zs[0], exp_z[0]), min(zs[-1], exp_z[-1])
        m = (zs >= lo) & (zs <= hi)
        exp_i = np_.stack(
            [np_.interp(zs[m], exp_z, exp[:, j]) for j in range(exp.shape[1])],
            axis=1,
        )
        np_.testing.assert_allclose(
            arr[m], exp_i, rtol=rtol, atol=atol,
            err_msg=(f"{name}: max|delta|={np_.abs(arr[m] - exp_i).max():.3e} "
                     f"over {m.sum()}/{len(zs)} z-samples"),
        )

    def flush(self):
        if self._mode == "record" and self._new:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np_.savez_compressed(self._path, **self._new)


@pytest.fixture
def golden(request):
    node = request.node.nodeid                       # tests/integration/test_x.py::test_y
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", node.replace("::", "__"))
    safe = safe.replace("tests_integration_", "").replace(".py", "")
    g = _Golden(GOLDEN_DIR / f"{safe}.npz", GOLDEN_MODE)
    yield g
    g.flush()
