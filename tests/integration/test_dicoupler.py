"""Integration test rewritten from ``docs_source/dc.rst`` (directional coupler)."""

import numpy as np
import pytest

from cbeam import waveguide, propagator

pytestmark = pytest.mark.integration

# ---- doc parameters ------------------------------------------------------- #
WL = 1.55
DMIN = 10.0
DMAX = 60.0
COUPLING_LENGTH = 5000.0
BEND_LENGTH = COUPLING_LENGTH / 4.0
RCORE = 3.0
NCLAD = 1.444
NCORE = NCLAD + 8.8e-3
CORE_RES = 15
CLAD_MESH_SIZE = 20.0
CORE_MESH_SIZE = 1.0


@pytest.fixture(scope="module")
def dicoupler():
    return waveguide.Dicoupler(
        RCORE, RCORE, NCORE, NCORE, DMAX, DMIN, NCLAD,
        COUPLING_LENGTH, BEND_LENGTH, CORE_RES,
        core_mesh_size=CORE_MESH_SIZE, clad_mesh_size=CLAD_MESH_SIZE,
    )


def test_length_is_auto_computed(dicoupler):
    assert dicoupler.z_ex == 2 * COUPLING_LENGTH


def test_plot_paths_and_mesh(dicoupler):
    dicoupler.plot_paths()
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    dicoupler.plot_mesh(z=0, ax=axs[0])
    dicoupler.plot_mesh(z=dicoupler.z_ex / 2, ax=axs[1])


def test_mode_solve_midpoint(dicoupler, save_dir):
    dc_prop = propagator.Propagator(WL, dicoupler, Nmax=2, save_dir=save_dir)
    neff, modes = dc_prop.solve_at(z=dicoupler.z_ex / 2)
    assert len(neff) == 2
    # in the coupling region the symmetric / antisymmetric split is resolved
    assert neff[0] > neff[1]
    dc_prop.plot_cfield(modes[1].astype(complex), mesh=dc_prop.mesh)


@pytest.mark.slow
def test_characterize_and_propagate(dicoupler, save_dir):
    dc_prop = propagator.Propagator(WL, dicoupler, Nmax=2, save_dir=save_dir)
    dc_prop.z_acc = -1.0
    zs, neffs, vs, cmats = dc_prop.characterize(save=False, tag="test_dicoupler")

    assert neffs.shape[1] == 2
    dc_prop.plot_neffs()
    dc_prop.plot_coupling_coeffs()

    u0 = [1, 0]
    zs, us, uf = dc_prop.propagate(u0)
    dc_prop.plot_mode_powers(zs, us)

    # power is conserved and light has coupled out of the launch channel
    assert np.sum(np.abs(uf) ** 2) == pytest.approx(1.0, abs=1e-3)
    powers = np.abs(us) ** 2
    assert powers[:, 1].max() > 0.1        # some power reached channel 2
    assert powers.sum(axis=1) == pytest.approx(np.ones(len(zs)), abs=1e-2)


@pytest.mark.slow
def test_vary_coupling_length_via_z_rescale(dicoupler, save_dir):
    """The doc reuses one characterization and stretches the z-array to sweep
    the coupling length.  We run a handful of stretches and check every
    propagation still conserves power."""
    dc_prop = propagator.Propagator(WL, dicoupler, Nmax=2, save_dir=save_dir)
    dc_prop.z_acc = -1.0
    dc_prop.characterize(save=False)

    u0 = [1, 0]
    for stretch in np.linspace(0, 10000, 6):
        zs = np.copy(dc_prop.zs)
        zs[np.argmax(zs >= dicoupler.z_ex / 2):] += stretch
        dc_prop.make_interp_funcs(zs)
        zs, us, uf = dc_prop.propagate(u0, zs[0], zs[-1])
        assert np.sum(np.abs(uf) ** 2) == pytest.approx(1.0, abs=2e-3)

    # restore the un-stretched interpolants
    dc_prop.make_interp_funcs()
