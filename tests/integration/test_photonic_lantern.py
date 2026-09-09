"""Integration test rewritten from ``docs_source/PL.rst`` (6-port photonic lantern).

The documentation runs the full 40 mm device (~80 s).  To keep CI sane the
slow tests here characterize a shorter leading section - the code path is
identical - and assert on physical invariants (power conservation, mode
structure) rather than on the exact numbers in the doc's ``testoutput``.
"""

import numpy as np
import pytest

from cbeam.waveguide import PhotonicLantern
from cbeam.propagator import Propagator

pytestmark = pytest.mark.integration

WL = 1.55


@pytest.fixture(scope="module")
def lantern():
    taper_factor = 8.0
    rcore = 2.2 / taper_factor
    rclad = 10
    rjack = 30
    z_ex = 40000
    nclad = 1.444
    ncore = nclad + 8.8e-3
    njack = nclad - 5.5e-3
    t = 2 * np.pi / 5
    core_offset = rclad * 2 / 3
    core_pos = np.array(
        [[0, 0]] + [[core_offset * np.cos(i * t), core_offset * np.sin(i * t)] for i in range(5)]
    )
    return PhotonicLantern(
        core_pos, [rcore] * 6, rclad, rjack, [ncore] * 6, nclad, njack,
        z_ex, taper_factor,
    )


def test_lantern_structure(lantern):
    assert lantern.linear is True
    assert lantern.z_ex == 40000
    d = lantern.assign_IOR()
    assert sum(k.startswith("core") for k in d) == 6


@pytest.mark.slow
def test_solve_at_midpoint(lantern, save_dir, golden):
    prop = Propagator(WL, lantern, 6, save_dir=save_dir)
    neff, modes = prop.solve_at(z=lantern.z_ex / 2.0)
    nclad, ncore, njack = 1.444, 1.444 + 8.8e-3, 1.444 - 5.5e-3
    assert len(neff) == 6
    assert np.all(np.diff(neff) <= 1e-9)          # index-ordered
    golden.check("neff", neff, rtol=1e-6, atol=1e-9)
    # every tracked mode sits inside the waveguide index range (not radiating)
    assert np.all(neff > njack)
    assert np.all(neff < ncore)
    # the fundamental is still a well-guided (core) mode mid-taper
    assert neff[0] > nclad
    prop.plot_cfield(modes[0].astype(complex), z=lantern.z_ex / 2.0)


@pytest.mark.slow
def test_characterize_propagate_and_channel_powers(lantern, save_dir, golden):
    prop = Propagator(WL, lantern, 6, save_dir=save_dir)
    prop.z_acc = -1.0
    # the full device: the isolated single-mode cores only become well separated
    # towards the back of the taper, so the channel-power extraction below is
    # only meaningful over the whole length (this is the doc's example verbatim)
    zs, neffs, vs, cmats = prop.characterize(save=False, tag="test")

    assert neffs.shape[1] == 6
    assert cmats.shape[1:] == (6, 6)
    assert np.isfinite(cmats).all()
    golden.check_vs_z("neffs", zs, neffs)

    prop.plot_neffs()
    prop.plot_coupling_coeffs()

    u0 = [1, 0, 0, 0, 0, 0]                       # launch LP01
    zs, us, uf = prop.propagate(u0)
    prop.plot_mode_powers(zs, us)
    assert np.sum(np.abs(uf) ** 2) == pytest.approx(1.0, abs=5e-3)

    amps = prop.to_channel_basis(uf)
    powers = np.abs(amps) ** 2
    assert len(powers) == 6
    assert np.isfinite(powers).all()
    # doc testoutput: [0.4957, 0.1006, 0.1004, 0.1004, 0.1005, 0.1006]
    # (exact split is mesh/degeneracy dependent; the pattern is robust)
    assert powers.sum() == pytest.approx(1.0, abs=0.1)
    assert powers[0] == pytest.approx(0.5, abs=0.15)
    assert powers[1:].min() > 0.03
    assert powers[0] > powers[1:].max()
    # the lantern has 5-fold rotational symmetry, so a centered LP01 launch should
    # split ~evenly across the 5 outer ports; a lopsided split can still pass the
    # loose bounds above but would indicate a symmetry bug
    outer = powers[1:]
    assert np.ptp(outer) / outer.mean() < 0.15
    # the 6 cores are symmetric, so which core carries which power can permute
    # between eigensolver runs -> compare the sorted power spectrum
    golden.check("channel_powers", powers, sort=True, atol=2e-3)
    golden.check("uf_total_power", np.sum(np.abs(uf) ** 2), atol=2e-3)


@pytest.mark.slow
def test_degen_groups_run(lantern, save_dir, golden):
    """The doc fixes the degenerate eigenbasis via ``degen_groups`` and shows the
    coupling coefficients shrink.  We just check the calculation still runs and
    conserves power."""
    prop = Propagator(WL, lantern, 6, save_dir=save_dir)
    prop.z_acc = -1.0
    prop.degen_groups = [[1, 2], [3, 4]]
    zs, neffs, vs, cmats = prop.characterize(0, 6000, save=False, tag="test_degen")

    # within a degenerate pair the effective indices are forced equal
    assert np.allclose(neffs[:, 1], neffs[:, 2])
    assert np.allclose(neffs[:, 3], neffs[:, 4])
    golden.check_vs_z("neffs", zs, neffs)

    zs, us, uf = prop.propagate([1, 0, 0, 0, 0, 0])
    assert np.sum(np.abs(uf) ** 2) == pytest.approx(1.0, abs=5e-3)
    golden.check("uf_power", np.abs(uf) ** 2, sort=True, atol=2e-3)
