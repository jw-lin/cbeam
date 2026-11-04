"""Smoke tests for ``examples/multi_wvl``.

Fast only: wavelength scaling, degenerate-multiplet splitting and the spectral
interpolation. A full
multi-wavelength propagation needs one lantern characterization per
wavelength (minutes each), which ``test_batch_pipeline.py`` already covers
for a single wavelength.
"""

import pathlib
import sys

import numpy as np
import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import batch_pipeline as bp  # noqa: E402
import multi_wvl as mw  # noqa: E402
from multi_wvl.bookkeeping import _multiplets  # noqa: E402


# --------------------------------------------------------------------------- #
# multi_wvl
# --------------------------------------------------------------------------- #
def test_scale_params_to_wavelength():
    base = bp.get_simulation_parameters(3, 0.8)
    p = mw.scale_params_to_wavelength(base, 900.0)
    assert p["wl"] == pytest.approx(0.9) and p["wavelength_nm"] == 900.0
    assert p["pixel_scale_um"] == pytest.approx(base["pixel_scale_um"] * 900 / 800)
    for k in ("nclad", "ncore", "njack", "rclad", "z_ex", "taper_factor"):
        assert p[k] == base[k]
    assert base["wl"] == pytest.approx(0.8)          # base left untouched


def test_multiplets():
    neff = np.array([1.450, 1.4499999, 1.449, 1.448, 1.44799995, 1.44799990, 1.447])
    assert _multiplets(neff, 1e-5) == [[0, 1], [2], [3, 4, 5], [6]]


def test_interpolate_complex_spectra():
    rng = np.random.default_rng(0)
    wl = np.array([700.0, 750.0, 800.0])
    spec = rng.normal(size=(3, 2, 19)) + 1j * rng.normal(size=(3, 2, 19))
    interp = mw.MultiWavelengthPropagationPipeline.interpolate_complex_spectra

    # reproduces the native samples, and interpolates the amplitude linearly
    np.testing.assert_allclose(interp(spec, wl, wl), spec, atol=1e-12)
    mid = interp(spec, wl, np.array([725.0]))[0]
    np.testing.assert_allclose(np.abs(mid), 0.5 * (np.abs(spec[0]) + np.abs(spec[1])))

    # an arbitrary per-wavelength phase (independent eigensolves) does not
    # change the interpolated power
    gauge = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(3, 1, 1)))
    wl_out = np.linspace(700, 800, 11)
    np.testing.assert_allclose(np.abs(interp(spec * gauge, wl, wl_out)),
                               np.abs(interp(spec, wl, wl_out)), rtol=1e-12)
