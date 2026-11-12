"""Integration tests for ``cbeam.chromatic`` on the directional coupler of
``docs_source/dc.rst`` (about 10 s per characterization)."""

import numpy as np
import pytest

from cbeam import waveguide, propagator, chromatic

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# ---- dc.rst parameters ------------------------------------------------------- #
WL_REF = 1.55
DMIN, DMAX = 10.0, 60.0
COUPLING_LENGTH = 5000.0
BEND_LENGTH = COUPLING_LENGTH / 4.0
RCORE = 3.0
NCLAD = 1.444
NCORE = NCLAD + 8.8e-3
CORE_RES = 15
CLAD_MESH_SIZE = 20.0
CORE_MESH_SIZE = 1.0


def make_dicoupler(nclad=NCLAD, ncore=NCORE):
    return waveguide.Dicoupler(
        RCORE, RCORE, ncore, ncore, DMAX, DMIN, nclad,
        COUPLING_LENGTH, BEND_LENGTH, CORE_RES,
        core_mesh_size=CORE_MESH_SIZE, clad_mesh_size=CLAD_MESH_SIZE,
    )


def make_build(save_dir, dispersive=False, log=None):
    """build(wl) for chromatic.sweep: fixed or silica-dispersive indices."""
    def build(wl):
        if dispersive:
            nclad, ncore = chromatic.dispersive_index([NCLAD, NCORE], wl, WL_REF)
            tag = chromatic.wavelength_tag("dc_silica", wl)
        else:
            nclad, ncore = NCLAD, NCORE
            tag = chromatic.wavelength_tag("dc_fixed", wl)
        prop = propagator.Propagator(wl, make_dicoupler(nclad, ncore), Nmax=2, save_dir=save_dir)
        prop.z_acc = -1.0
        characterized = chromatic.load_or_characterize(prop, tag)
        if log is not None:
            log.append(characterized)
        return prop
    return build


def test_sweep_fixed_indices(save_dir, golden):
    wls = [1.45, 1.55, 1.65]
    res = chromatic.sweep(make_build(save_dir), wls, verbose=False)

    assert res["transfer_matrix"].shape == (3, 2, 2)
    power = np.abs(res["transfer_matrix"]) ** 2
    # lossless 2x2 coupler: every input's power reaches the two channels
    np.testing.assert_allclose(power.sum(axis=1), 1.0, atol=1e-2)
    cross = power[:, 1, 0]
    # the splitting ratio is chromatic: it changes substantially over the band
    assert cross.max() - cross.min() > 0.3
    # longer wavelengths: lower effective indices
    assert np.all(np.diff(res["neffs_in"].max(axis=1)) < 0)
    golden.check("cross_power", cross, atol=1e-3)
    golden.check("neffs_in", np.sort(res["neffs_in"], axis=1), rtol=1e-6, atol=1e-8)


def test_characterizations_are_cached(save_dir):
    log = []
    build = make_build(save_dir, log=log)
    first = chromatic.sweep(build, [1.5], verbose=False)
    second = chromatic.sweep(build, [1.5], verbose=False)
    assert log == [True, False]              # characterized once, then loaded
    np.testing.assert_allclose(np.abs(second["transfer_matrix"]) ** 2,
                               np.abs(first["transfer_matrix"]) ** 2, atol=1e-6)


def test_chain_matches_single_propagator(save_dir):
    """A ChainPropagator split at the middle of the coupler gives the same
    channel powers as one Propagator over the whole length."""
    def build_chain(wl):
        wvg = make_dicoupler()
        front = propagator.Propagator(wl, wvg, Nmax=2, save_dir=save_dir)
        back = propagator.Propagator(wl, wvg, Nmax=2, save_dir=save_dir)
        front.z_acc = back.z_acc = -1.0
        z_mid = wvg.z_ex / 2
        chromatic.load_or_characterize(front, chromatic.wavelength_tag("dc_front", wl), 0.0, z_mid)
        back.load_init_conds(front)
        chromatic.load_or_characterize(back, chromatic.wavelength_tag("dc_back", wl), z_mid, wvg.z_ex)
        return propagator.ChainPropagator([front, back])

    chain = chromatic.sweep(build_chain, [WL_REF], verbose=False)
    single = chromatic.sweep(make_build(save_dir), [WL_REF], verbose=False)
    np.testing.assert_allclose(np.abs(chain["transfer_matrix"]) ** 2,
                               np.abs(single["transfer_matrix"]) ** 2, atol=2e-2)
    np.testing.assert_allclose(chain["neffs_in"], single["neffs_in"], atol=1e-6)


def test_dispersive_indices(save_dir):
    """Silica dispersion lowers every index by the same amount at longer
    wavelengths; at the reference wavelength nothing changes."""
    wls = [WL_REF, 1.65]
    fixed = chromatic.sweep(make_build(save_dir), wls, verbose=False)
    silica = chromatic.sweep(make_build(save_dir, dispersive=True), wls, verbose=False)
    np.testing.assert_allclose(silica["neffs_in"][0], fixed["neffs_in"][0], atol=1e-6)
    shift = chromatic.silica_index(1.65) - chromatic.silica_index(WL_REF)
    # weakly guiding: the modes follow the common index shift (2e-6 off on this mesh)
    np.testing.assert_allclose(silica["neffs_in"][1] - fixed["neffs_in"][1], shift, atol=2e-5)
    power = np.abs(silica["transfer_matrix"]) ** 2
    np.testing.assert_allclose(power.sum(axis=1), 1.0, atol=1e-2)
