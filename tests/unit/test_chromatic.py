"""Unit tests for ``cbeam.chromatic``: no characterization, stand-in propagators."""

import numpy as np
import pytest

from cbeam import chromatic


# --------------------------------------------------------------------------- #
# refractive-index models
# --------------------------------------------------------------------------- #
def test_silica_index_sodium_d_line():
    # fused silica n_d = 1.4585 at 587.6 nm (Malitson 1965)
    assert chromatic.silica_index(0.5876) == pytest.approx(1.4585, abs=1e-4)


def test_silica_index_normal_dispersion():
    n = chromatic.silica_index(np.linspace(0.5, 1.6, 12))
    assert n.shape == (12,)
    assert np.all(np.diff(n) < 0)


def test_dispersive_index_reference_and_contrast():
    nclad, ncore = 1.444, 1.444 + 8.8e-3
    assert chromatic.dispersive_index(nclad, 1.55, 1.55) == pytest.approx(nclad)
    for wl in (0.8, 1.31, 1.65):
        n_cl = chromatic.dispersive_index(nclad, wl, 1.55)
        n_co = chromatic.dispersive_index(ncore, wl, 1.55)
        assert n_co - n_cl == pytest.approx(ncore - nclad)          # contrast kept
        assert n_cl - nclad == pytest.approx(
            chromatic.silica_index(wl) - chromatic.silica_index(1.55))
    # arrays of indices go through in one call
    out = chromatic.dispersive_index([nclad, ncore], 0.8, 1.55)
    assert out.shape == (2,) and out[1] - out[0] == pytest.approx(ncore - nclad)


def test_dispersive_index_custom_material():
    flat = lambda wl: 1.5 + 0.0 * np.asarray(wl)
    assert chromatic.dispersive_index(1.45, 0.8, 1.55, material=flat) == pytest.approx(1.45)


def test_wavelength_tag():
    assert chromatic.wavelength_tag("dc", 1.55) == "dc_1550.0nm"
    assert chromatic.wavelength_tag("pl", 0.8125) == "pl_812.5nm"


# --------------------------------------------------------------------------- #
# stand-ins for Propagator / ChainPropagator
# --------------------------------------------------------------------------- #
class FakeProp:
    """Propagator-like: neffs over z, a transfer matrix, load / characterize."""
    xp = np

    def __init__(self, wl, saved=()):
        self.wl, self.saved, self.calls = wl, set(saved), []
        self.neffs = np.array([[1.45 - 0.01 * wl, 1.44 - 0.01 * wl],
                               [1.46 - 0.01 * wl, 1.43 - 0.01 * wl]])

    def load(self, tag):
        self.calls.append(("load", tag))
        if tag not in self.saved:
            raise FileNotFoundError(tag)

    def characterize(self, zi=None, zf=None, save=False, tag=""):
        self.calls.append(("characterize", zi, zf, save, tag))

    def compute_transfer_matrix(self, channel_basis=True):
        return np.array([[np.cos(self.wl), 1j * np.sin(self.wl)],
                         [1j * np.sin(self.wl), np.cos(self.wl)]])


class FakeChain:
    """ChainPropagator-like: no compute_transfer_matrix, propagate + to_channel_basis."""
    xp = np
    Nmax = 3

    def __init__(self, M, skipped):
        self.M = np.asarray(M, dtype=complex)
        self.propagators = [FakeProp(1.0), FakeProp(1.0)]
        self.propagators[0].skipped_modes = set(skipped)

    def propagate(self, u0):
        return None, None, self.M @ u0

    def to_channel_basis(self, uf, z=None):
        return uf[::-1]                       # a fixed, recognisable change of basis


# --------------------------------------------------------------------------- #
# load_or_characterize
# --------------------------------------------------------------------------- #
def test_load_or_characterize_characterizes_when_missing():
    prop = FakeProp(1.55)
    assert chromatic.load_or_characterize(prop, "t", zi=0.0, zf=10.0) is True
    assert prop.calls == [("load", "t"), ("characterize", 0.0, 10.0, True, "t")]


def test_load_or_characterize_loads_when_saved():
    prop = FakeProp(1.55, saved={"t"})
    assert chromatic.load_or_characterize(prop, "t") is False
    assert prop.calls == [("load", "t")]


# --------------------------------------------------------------------------- #
# measures
# --------------------------------------------------------------------------- #
def test_measures_on_a_propagator():
    prop = FakeProp(1.2)
    np.testing.assert_allclose(chromatic.neffs_in(prop), prop.neffs[0])
    np.testing.assert_allclose(chromatic.neffs_out(prop), prop.neffs[-1])
    np.testing.assert_allclose(chromatic.transfer_matrix(prop), prop.compute_transfer_matrix())


def test_measures_on_a_chain():
    M = np.arange(9).reshape(3, 3) + 1j
    chain = FakeChain(M, skipped={1})
    chain.propagators[0].neffs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    chain.propagators[1].neffs = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    np.testing.assert_allclose(chromatic.neffs_in(chain), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(chromatic.neffs_out(chain), [10.0, 11.0, 12.0])

    T = chromatic.transfer_matrix(chain, channel_basis=True)
    expected = M[::-1].copy()
    expected[:, 1] = 0                       # skipped input mode -> zero column
    np.testing.assert_allclose(T, expected)
    T_eig = chromatic.transfer_matrix(chain, channel_basis=False)
    np.testing.assert_allclose(T_eig[:, [0, 2]], M[:, [0, 2]])


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def test_sweep_default_measures():
    built = []

    def build(wl):
        built.append(wl)
        return FakeProp(wl)

    wls = [1.45, 1.55, 1.65]
    res = chromatic.sweep(build, wls, verbose=False)
    assert built == wls                                   # one build per wavelength, in order
    np.testing.assert_allclose(res["wavelengths"], wls)
    assert res["transfer_matrix"].shape == (3, 2, 2)
    assert res["neffs_in"].shape == res["neffs_out"].shape == (3, 2)
    for i, wl in enumerate(wls):
        np.testing.assert_allclose(res["transfer_matrix"][i], FakeProp(wl).compute_transfer_matrix())
        np.testing.assert_allclose(res["neffs_out"][i], FakeProp(wl).neffs[-1])


def test_sweep_custom_measures_and_generator_input():
    res = chromatic.sweep(lambda wl: FakeProp(wl), (wl for wl in (1.0, 2.0)),
                          measures={"wl": lambda p: p.wl,
                                    "cross": lambda p: np.abs(chromatic.transfer_matrix(p)[1, 0]) ** 2},
                          verbose=False)
    assert set(res) == {"wl", "cross", "wavelengths"}
    np.testing.assert_allclose(res["wl"], [1.0, 2.0])
    np.testing.assert_allclose(res["cross"], np.sin([1.0, 2.0]) ** 2)
