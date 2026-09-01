"""Integration test rewritten from ``docs_source/PL19.rst`` (19-port lantern).

The real example is a ~100 mm device that takes ~13 min to characterize in two
halves.  These tests build the same waveguide, run the (cheaper) effective-index
scan the doc uses to plan the calculation, and exercise the ``ChainPropagator``
end-to-end wiring on a heavily shortened device.  All are ``slow``.
"""

import numpy as np
import pytest

from cbeam.waveguide import get_19port_positions, PhotonicLantern
from cbeam.propagator import Propagator, ChainPropagator

pytestmark = [pytest.mark.integration, pytest.mark.slow]

WL = 0.8


def _make_pl19(z_ex):
    taper_factor = 12.0
    rcore = 1.8 / taper_factor
    rclad = 9.0
    rjack = 27
    nclad = 1.444
    ncore = nclad + 8.8e-3
    njack = nclad - 5.5e-3
    core_pos = get_19port_positions(rclad / 2.5)
    # coarser meshing than the doc (which leaves the defaults) so the 19 tiny
    # cores don't blow the mesh up to tens of thousands of points
    return PhotonicLantern(
        core_pos, [rcore] * 19, rclad, rjack, [ncore] * 19, nclad, njack,
        z_ex, taper_factor, core_res=12, clad_res=40, jack_res=24,
        core_mesh_size=0.5, clad_mesh_size=3.0,
    )


def test_19port_positions_hex_layout():
    core_pos = get_19port_positions(9.0 / 2.5)
    assert core_pos.shape == (19, 2)
    assert np.allclose(core_pos[0], [0, 0])


def test_compute_neffs_scan(save_dir):
    pl19 = _make_pl19(z_ex=10000)          # doc uses 100000
    prop = Propagator(WL, pl19, Nmax=21, save_dir=save_dir)
    prop.z_acc = -1.0
    zs, neffs = prop.compute_neffs(save=False)

    assert neffs.shape == (len(zs), 21)
    assert np.all(np.isfinite(neffs))
    # modes are index-ordered at the front of the device
    assert np.all(np.diff(neffs[0]) <= 1e-9)
    prop.plot_neff_diffs()


def test_chain_propagator_end_to_end(save_dir):
    pl19 = _make_pl19(z_ex=8000)
    half = pl19.z_ex / 2

    prop1 = Propagator(WL, pl19, 20, save_dir=save_dir + "/a")
    prop1.z_acc = -1.0
    prop1.degen_groups = [[1, 2], [3, 4], [6, 7], [8, 9], [10, 11], [12, 13], [15, 16]]
    prop1.skipped_modes = [18]
    prop1.characterize(0, half, save=False, tag="pl19_front")

    prop2 = Propagator(WL, pl19, 20, save_dir=save_dir + "/b")
    prop2.z_acc = -1.0
    prop2.skipped_modes = [18]
    prop2.degen_groups = [[i for i in range(20)]]
    del prop2.degen_groups[0][18]
    prop2.load_init_conds(prop1)
    prop2.characterize(half, pl19.z_ex, save=False, tag="pl19_back")

    chain = ChainPropagator([prop1, prop2])
    u0 = [0.0] * 20
    u0[0] = 1.0
    zs, us, uf = chain.propagate(u0)
    chain.plot_mode_powers(zs, us)

    assert us.shape[1] == 20
    # mode 18 is skipped/zeroed throughout
    assert np.allclose(us[:, 18], 0.0)
    # power (minus the skipped mode) is essentially conserved
    assert np.sum(np.abs(uf) ** 2) == pytest.approx(1.0, abs=0.05)

    out = chain.to_channel_basis(uf)
    assert len(out) == 19
