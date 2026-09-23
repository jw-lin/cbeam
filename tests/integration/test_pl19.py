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


def test_compute_neffs_scan(save_dir, golden):
    pl19 = _make_pl19(z_ex=10000)          # doc uses 100000
    prop = Propagator(WL, pl19, Nmax=21, save_dir=save_dir)
    prop.z_acc = -1.0
    zs, neffs = prop.compute_neffs(save=False)

    assert neffs.shape == (len(zs), 21)
    assert np.all(np.isfinite(neffs))
    # modes are index-ordered at the front of the device
    assert np.all(np.diff(neffs[0]) <= 1e-9)
    golden.check_vs_z("neffs", zs, neffs, rtol=1e-4, atol=1e-6)
    prop.plot_neff_diffs()


def test_chain_propagator_end_to_end(save_dir, golden):
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

    # the trajectory must reach the true end of the second segment, not get
    # stuck reporting the front/back junction (half) as the last point --
    # regression check for a bug where the second segment's own trajectory
    # was silently dropped entirely on the jax backend.
    assert float(zs[-1]) == pytest.approx(pl19.z_ex)
    assert us.shape[1] == 20
    # mode 18 is skipped/zeroed throughout
    assert np.allclose(us[:, 18], 0.0)
    # power (minus the skipped mode) is essentially conserved
    assert np.sum(np.abs(uf) ** 2) == pytest.approx(1.0, abs=0.05)

    out = chain.to_channel_basis(uf)
    assert len(out) == 19
    # The 19 modes here live in one ~19-fold degenerate subspace, so the
    # per-mode power split (uf) is gauge-dependent and not a stable fingerprint
    # (it moves by ~5e-2 between eigensolver bases).  The physical channel-power
    # spectrum is stable, so only that is pinned.
    golden.check("channel_out_power", np.abs(out) ** 2, sort=True, atol=1e-3)

    # n_save asks for a dense, fixed-size trajectory instead of just
    # endpoints -- needed on the jax backend for plotting (see
    # Propagator.propagate()'s docstring); must reach the same final state
    # as the default endpoint-only call, on a real (not hand-built) chain.
    zs_dense, us_dense, uf_dense = chain.propagate(u0, n_save=25)
    assert float(zs_dense[0]) == pytest.approx(0.0)
    assert float(zs_dense[-1]) == pytest.approx(pl19.z_ex)
    assert np.allclose(uf_dense, uf, atol=1e-6)

    # --- field reconstruction on a chain -------------------------------
    # ChainPropagator.make_field() dispatches to whichever segment owns z.
    # zs[-1] is the far end of the *last* segment, i.e. exactly the
    # boundary that get_prop() has to resolve to prop2 rather than prop1.
    assert chain.get_prop(float(zs[-1])) is prop2
    f = chain.make_field(us[-1], zs[-1])
    assert f.shape[0] == prop2.mesh.points.shape[0]
    assert np.isfinite(np.asarray(f)).all()
    assert np.allclose(f, prop2.make_field(us[-1], zs[-1]))
    # the reconstructed field is the *physical* field: the 19 tracked modes
    # form one degenerate group, so the mode amplitudes us[-1] and the basis
    # they multiply are both gauge-dependent, but their sum is not (up to a
    # global phase).  So |f| is comparable across runs and backends, and is
    # what pins this reconstruction.  numpy and jax were measured to agree
    # here to ~3e-9; the 1e-3 budget is deliberate slack for the eigensolver
    # nondeterminism documented in Propagator.compute_modes().
    fabs = np.abs(np.asarray(f))
    assert fabs.max() > 0
    golden.check("field_abs", fabs, rtol=1e-3, atol=1e-3 * float(fabs.max()))
    chain.plot_cfield(f, zs[-1], res=2.0, show_mesh=True,
                      xlim=(-100, 100), ylim=(-100, 100))

    # --- transfer matrix on a chain ------------------------------------
    # compute_transfer_matrix() is inherited from Propagator and never
    # overridden, so on a chain it relies on ChainPropagator.propagate() and
    # .to_channel_basis() to resolve zi=zf=None to the ends of the *whole*
    # chain themselves -- nothing else exercises that path.
    M = np.asarray(chain.compute_transfer_matrix())
    assert M.shape == (20, 20)
    assert np.isfinite(M).all()
    assert np.allclose(M[:, 18], 0.0)            # the skipped mode
    # M is padded from 19 channels out to Nmax=20 rows
    assert np.allclose(M[19:, :], 0.0)
    # the point of the matrix: propagation *is* M @ u0.  Column 0 must
    # therefore reproduce, exactly, the channel amplitudes propagate() +
    # to_channel_basis() gave for the same LP01 launch above.
    out_via_M = M @ np.asarray(u0)
    assert np.allclose(out_via_M[:19], np.asarray(out), atol=1e-8)
    # each non-skipped input mode carries unit power into the channel basis
    col_powers = np.sum(np.abs(M) ** 2, axis=0)
    kept = [j for j in range(20) if j != 18]
    assert np.allclose(col_powers[kept], 1.0, atol=0.05)
    # M's *input* index is the mode basis, which is gauge-dependent here, but
    # a change of that basis is a unitary acting on M's columns -- so the
    # singular values are invariant and can be pinned.  They come out just
    # under 1 (propagation is unitary, the channel basis very nearly
    # orthonormal) with a single exact zero for the skipped mode; numpy and
    # jax were measured to agree on them to ~1e-8.
    svals = np.linalg.svd(M, compute_uv=False)
    assert svals[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(svals[:19], 1.0, atol=0.05)
    golden.check("transfer_svals", svals, rtol=1e-3, atol=1e-3)
