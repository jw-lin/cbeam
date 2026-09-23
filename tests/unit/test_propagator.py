"""Unit tests for ``cbeam.propagator`` that do not need a full characterization.

The genuinely expensive code paths (adaptive ``compute_modes`` /
``compute_cmats`` on a tapering waveguide) are covered in
``tests/integration``.  Here we test the numerical helpers directly, plus the
cheap *z-invariant* code path end to end.
"""

import os

import numpy as np
import pytest
import scipy.sparse as sp

from cbeam import waveguide as wg
from cbeam.propagator import Propagator, ChainPropagator


@pytest.fixture
def fiber():
    return wg.CircularStepIndexFiber(10, 30, 1.445, 1.44, core_res=16, clad_res=32)


@pytest.fixture
def prop(fiber, save_dir):
    return Propagator(1.55, fiber, Nmax=4, save_dir=save_dir)


# --------------------------------------------------------------------------- #
# construction / io
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_wavenumber(self, prop):
        assert prop.k == pytest.approx(2 * np.pi / 1.55)

    def test_wl_is_read_only(self, prop):
        # A Propagator's mode data, coupling matrices, and (on the jax
        # backend) cached ODE-step function are all specific to the
        # wavelength it was characterized at -- self.k, and anything built
        # from it, would go stale if .wl could change after construction.
        # The multi-wavelength pipeline already only ever constructs a
        # fresh Propagator per wavelength (see e.g.
        # examples/multi_wvl/pipeline.py), so this formalizes an invariant
        # nothing relies on being able to break.
        with pytest.raises(AttributeError):
            prop.wl = 1.31

    def test_k_is_derived_from_wl(self, prop):
        assert prop.k == pytest.approx(2 * np.pi / prop.wl)

    def test_creates_output_folders(self, fiber, save_dir):
        Propagator(1.55, fiber, Nmax=4, save_dir=save_dir)
        for sub in ("eigenmodes", "eigenvalues", "cplcoeffs", "zvals",
                    "meshes", "meshpoints"):
            assert os.path.isdir(os.path.join(save_dir, sub))

    def test_default_save_dir_is_local_data(self, fiber, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = Propagator(1.55, fiber, Nmax=2)
        assert p.save_dir == "./data"
        assert (tmp_path / "data" / "eigenmodes").is_dir()


# --------------------------------------------------------------------------- #
# pure numerical helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_inner_product_is_v1T_B_v2(self, prop):
        rng = np.random.default_rng(0)
        v1 = rng.normal(size=(3, 20))
        v2 = rng.normal(size=(3, 20))
        B = sp.random(20, 20, density=0.2, random_state=1)
        B = (B + B.T).tocsr()
        out = prop.inner_product(v1, v2, B)
        assert out.shape == (3, 3)
        assert np.allclose(out, v1 @ (B @ v2.T))

    def test_inner_product_self_is_norm_squared(self, prop):
        v = np.array([[1.0, 2.0, 3.0, 4.0]])
        B = sp.identity(4)
        assert prop.inner_product(v, v, B) == pytest.approx(30.0)

    def test_decimate_averages_down_to_ten(self, prop):
        arr = np.ones((4, 137))
        out = prop.decimate(arr)
        assert out.shape == (4, 10)
        assert np.allclose(out, 1.0)

    def test_swap_modes_applies_one_permutation_to_both(self, prop):
        # tag each eigenmode row with its original index so the permutation
        # can be recovered from the output
        w = np.array([2.0, 3.0, 1.0])
        _w = np.array([10.0, 20.0, 30.0])
        _v = np.array([[0.0], [1.0], [2.0]])
        v_out, w_out = prop.swap_modes(w, _w.copy(), _v.copy())
        perm = v_out[:, 0].astype(int)
        assert sorted(perm) == [0, 1, 2]                 # it is a permutation
        assert np.array_equal(w_out, _w[perm])           # same perm on w and v

    def test_swap_modes_is_identity_when_already_descending(self, prop):
        w = np.array([3.0, 2.0, 1.0])
        _w = np.array([3.1, 2.1, 1.1])
        _v = np.arange(9, dtype=float).reshape(3, 3)
        v_out, w_out = prop.swap_modes(w, _w.copy(), _v.copy())
        assert np.array_equal(w_out, _w)
        assert np.array_equal(v_out, _v)

    def test_make_sign_consistent_flips_opposite_modes(self, prop):
        rng = np.random.default_rng(2)
        v = rng.normal(size=(3, 15))
        _v = v.copy()
        _v[1] *= -1
        mask = prop.make_sign_consistent(v, _v)
        assert list(mask) == [False, True, False]
        assert np.allclose(_v, v)

    def test_correct_degeneracy_recovers_rotation(self, prop):
        rng = np.random.default_rng(3)
        # two orthonormal "degenerate" modes
        a = rng.normal(size=20)
        a /= np.linalg.norm(a)
        b = rng.normal(size=20)
        b -= b @ a * a
        b /= np.linalg.norm(b)
        v = np.stack([a, b])
        theta = 0.6
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        _v = rot @ v
        v_full = np.stack([a, b, rng.normal(size=20)])
        _v_full = np.stack([_v[0], _v[1], v_full[2]])
        prop.correct_degeneracy([0, 1], v_full, _v_full)
        assert np.allclose(np.abs(_v_full[:2]), np.abs(v_full[:2]), atol=1e-8)

    def test_avg_degen_neff(self, prop):
        neffs = np.array([1.45, 1.40, 1.42, 1.30])
        prop.avg_degen_neff([1, 2], neffs)
        assert neffs[1] == pytest.approx(1.41)
        assert neffs[2] == pytest.approx(1.41)

    def test_compute_isolated_basis_reports_both_failures(self, prop, monkeypatch):
        # If the requested z fails *and* the fallback (output end) also
        # fails, the final error must name both failures, not just the
        # fallback's -- a log or error tracker that captures only
        # str(exception), rather than the full chained traceback, would
        # otherwise lose which z the original failure was at.
        prop.zs = np.array([0.0, 100.0])
        attempted = []

        def always_fails(z):
            attempted.append(z)
            raise ValueError(f"boom at z={z}")

        monkeypatch.setattr(prop, "_compute_isolated_basis_at_z", always_fails)
        with pytest.raises(RuntimeError) as excinfo:
            prop.compute_isolated_basis(z=50.0)

        assert attempted == [50.0, 100.0]  # requested z, then the fallback
        msg = str(excinfo.value)
        assert "boom at z=50.0" in msg
        assert "boom at z=100.0" in msg
        assert isinstance(excinfo.value.__cause__, ValueError)


# --------------------------------------------------------------------------- #
# interpolation helpers
# --------------------------------------------------------------------------- #
class TestInterpolation:
    @pytest.fixture
    def loaded(self, prop):
        zs = np.linspace(0, 100, 6)
        prop.Nmax = 3
        prop.zs = zs
        prop.neffs = np.stack([
            1.45 - 1e-4 * zs,
            1.44 - 5e-5 * zs,
            1.43 + 2e-5 * zs,
        ], axis=1)
        prop.cmats = np.zeros((len(zs), 3, 3))
        prop.cmats[:, 0, 1] = 1e-3 * zs
        prop.cmats[:, 1, 0] = -1e-3 * zs
        prop.vs = np.zeros((len(zs), 3, 8))
        prop.make_interp_funcs()
        return prop

    def test_get_neff_interpolates(self, loaded):
        n = loaded.get_neff(0.0)
        assert n == pytest.approx(loaded.neffs[0], rel=1e-6)

    def test_get_cmat_antisymmetric(self, loaded):
        c = loaded.get_cmat(50.0)
        assert np.allclose(c, -c.T)
        assert c[0, 1] == pytest.approx(-c[1, 0])

    def test_int_neff_is_antiderivative(self, loaded):
        # d/dz of the integral == the value
        z = 40.0
        h = 1e-3
        approx = (loaded.get_int_neff(z + h) - loaded.get_int_neff(z - h)) / (2 * h)
        assert np.allclose(approx, loaded.get_neff(z), rtol=1e-4)

    def test_apply_phase_unit_modulus(self, loaded):
        u = np.array([1.0 + 0j, 0.5j, -0.3])
        out = loaded.apply_phase(u, 80.0, 0.0)
        assert np.allclose(np.abs(out), np.abs(u))

    def test_wkb_cor_shape(self, loaded):
        cor = loaded.WKB_cor(30.0)
        assert cor.shape == (3,)

    def test_propagate_uses_correct_zi_not_stale_cache(self, loaded):
        # On the jax backend, propagate()'s ODE step function is now built
        # once and reused across calls (see _get_jax_step_fn), with zi/zf
        # passed in as traced arguments instead of being closed over. Guard
        # against the regression that refactor could introduce: a cached
        # step function that silently keeps using the *first* zi it was
        # built with, regardless of what it's actually called with.
        u0 = [1.0, 0.0, 0.0]
        _, _, uf_a = loaded.propagate(u0, 10.0, 60.0)
        _, _, uf_b = loaded.propagate(u0, 20.0, 60.0)  # different zi, same zf
        assert not np.allclose(uf_a, uf_b, atol=1e-6)

        # and it must agree with a completely fresh (uncached) step function
        loaded.make_interp_funcs()
        _, _, uf_a_fresh = loaded.propagate(u0, 10.0, 60.0)
        assert np.allclose(uf_a, uf_a_fresh, atol=1e-8)

    def test_propagate_n_save_returns_dense_trajectory(self, loaded):
        # On the jax backend, propagate() normally keeps only the ODE
        # endpoint (SaveAt(t1=True)) to avoid buffering a per-step
        # trajectory -- fine for the physics, but useless for plotting the
        # trajectory itself (see ChainPropagator usage in
        # examples/original_example.ipynb, where this showed up as a
        # visibly flat/near-empty mode-power plot on jax). n_save opts back
        # into a dense, fixed-size trajectory on demand, on both backends.
        u0 = [1.0, 0.0, 0.0]
        zs, us, uf = loaded.propagate(u0, 10.0, 60.0, n_save=9)
        zs = np.asarray(zs)
        assert zs.shape[0] == 9
        assert np.asarray(us).shape[0] == 9
        assert float(zs[0]) == pytest.approx(10.0)
        assert float(zs[-1]) == pytest.approx(60.0)

        # physics must match the default (endpoint-only) call
        _, _, uf_default = loaded.propagate(u0, 10.0, 60.0)
        assert np.allclose(uf, uf_default, atol=1e-8)

    def test_jax_step_fn_is_cached_across_calls(self, loaded):
        u0 = [1.0, 0.0, 0.0]
        loaded.propagate(u0, 10.0, 60.0)
        if loaded.backend != "jax":
            pytest.skip("jax-only: propagate() has no compiled step "
                        "function to cache on the numpy backend")
        cached_fn = loaded._jax_step_cache[1]

        # same zi/zf: must reuse the compiled function, not rebuild it
        loaded.propagate(u0, 10.0, 60.0)
        assert loaded._jax_step_cache[1] is cached_fn

        # different zi/zf, same interpolants: still reused -- this is the
        # whole point (compute_transfer_matrix's per-mode loop, or repeated
        # propagate() calls at different z spans, used to force a fresh
        # jax.jit trace+compile every single time)
        loaded.propagate(u0, 20.0, 70.0)
        assert loaded._jax_step_cache[1] is cached_fn

        # backpropagate() shares the same cached step function
        loaded.backpropagate([0.0, 1.0, 0.0], 60.0, 10.0)
        assert loaded._jax_step_cache[1] is cached_fn

        # a real rebuild of the interpolants must invalidate the cache
        loaded.make_interp_funcs()
        loaded.propagate(u0, 10.0, 60.0)
        assert loaded._jax_step_cache[1] is not cached_fn


# --------------------------------------------------------------------------- #
# z-invariant end-to-end (cheap)
# --------------------------------------------------------------------------- #
class TestZInvariant:
    def test_solve_at_orders_modes_by_index(self, prop):
        neff, v = prop.solve_at(0)
        assert len(neff) == 4
        assert np.all(np.diff(neff) <= 1e-9)          # descending
        assert neff[0] < 1.445 and neff[0] > 1.44     # guided fundamental
        assert v.shape[0] == 4

    def test_characterize_zinv_returns_single_slice(self, prop):
        zs, neffs, vs, cmats = prop.characterize(save=False)
        assert zs.shape == (1,)
        assert neffs.shape == (1, 4)
        assert cmats is None

    def test_propagate_zinv_conserves_power(self, prop):
        prop.characterize(save=False)
        u0 = [1.0, 0.0, 0.0, 0.0]
        zs, us, uf = prop.propagate(u0, 0.0, 5000.0)
        assert np.sum(np.abs(uf) ** 2) == pytest.approx(1.0, rel=1e-6)

    def test_compute_transfer_matrix_matches_propagate_per_mode(self, prop):
        # compute_transfer_matrix() used to build its result with in-place
        # item assignment (mat[:M, j] = out, u0[j] = 1.), which crashes on
        # the jax backend since jax arrays are immutable. Regression test
        # for both backends: it must run at all, and every column must
        # equal what propagate() gives for that basis mode alone.
        prop.characterize(save=False)
        mat = prop.compute_transfer_matrix(channel_basis=False, zi=0.0, zf=5000.0)
        assert mat.shape == (prop.Nmax, prop.Nmax)
        for j in range(prop.Nmax):
            u0 = [0.0] * prop.Nmax
            u0[j] = 1.0
            _, _, uf = prop.propagate(u0, 0.0, 5000.0)
            assert np.allclose(mat[:, j], uf, atol=1e-10)
        # z-invariant fiber: no mode coupling, so propagation is phase-only
        # -> the matrix is diagonal with unit-modulus entries.
        off_diag = mat - np.diag(np.diag(mat))
        assert np.allclose(off_diag, 0, atol=1e-10)
        assert np.allclose(np.abs(np.diag(mat)), 1.0, rtol=1e-6)

    def test_compute_transfer_matrix_zeroes_skipped_modes(self, prop):
        prop.characterize(save=False)
        prop.skipped_modes = [1]
        mat = prop.compute_transfer_matrix(channel_basis=False, zi=0.0, zf=5000.0)
        assert np.allclose(mat[:, 1], 0.0)

    def test_make_field_and_back(self, prop):
        prop.characterize(save=False)
        field = prop.make_field([1.0, 0.0, 0.0, 0.0], z=0.0, apply_phase=False)
        assert field.shape[0] == prop.mesh.points.shape[0]
        amps = prop.make_mode_vector(field, z=0.0)
        assert np.argmax(np.abs(amps)) == 0
        assert np.abs(amps[0]) == pytest.approx(1.0, rel=1e-3)

    def test_save_and_load_roundtrip(self, prop):
        prop.characterize(save=True, tag="unit_rt")
        p2 = Propagator(1.55, prop.wvg, Nmax=4, save_dir=prop.save_dir)
        p2.load(tag="unit_rt")
        assert np.allclose(p2.neffs, prop.neffs)
        assert p2.zs.shape == prop.zs.shape

    def test_plot_helpers_run(self, prop):
        prop.characterize(save=False)
        prop.plot_cfield(prop.vs[0][0].astype(complex), z=0.0)


# --------------------------------------------------------------------------- #
# ChainPropagator wiring
# --------------------------------------------------------------------------- #
class TestChainPropagator:
    def test_concatenates_z_and_metadata(self, fiber, save_dir):
        p1 = Propagator(1.55, fiber, Nmax=3, save_dir=save_dir)
        p1.zs = np.linspace(0, 100, 5)
        p2 = Propagator(1.55, fiber, Nmax=3, save_dir=save_dir)
        p2.zs = np.linspace(100, 250, 4)
        chain = ChainPropagator([p1, p2])
        assert chain.z_breaks == [0.0, 100.0, 250.0]
        assert chain.zs.shape == (9,)
        assert chain.wl == 1.55
        assert chain.Nmax == 3

    def test_get_prop_selects_by_z(self, fiber, save_dir):
        p1 = Propagator(1.55, fiber, Nmax=3, save_dir=save_dir)
        p1.zs = np.linspace(0, 100, 5)
        p2 = Propagator(1.55, fiber, Nmax=3, save_dir=save_dir)
        p2.zs = np.linspace(100, 250, 4)
        chain = ChainPropagator([p1, p2])
        assert chain.get_prop(10.0) is p1
        assert chain.get_prop(200.0) is p2

    def test_wl_is_read_only(self, fiber, save_dir):
        p1 = Propagator(1.55, fiber, Nmax=3, save_dir=save_dir)
        p1.zs = np.linspace(0, 100, 5)
        chain = ChainPropagator([p1])
        with pytest.raises(AttributeError):
            chain.wl = 1.31

    def test_propagate_full_trajectory_across_segments(self, fiber, save_dir):
        # Regression test: propagate() assumed every segment's own
        # propagate() returns a multi-point trajectory whose first point
        # duplicates the previous segment's last point, dropping it via
        # zs[1:]/us[1:]. On the jax backend, each segment's propagate()
        # returns only the ODE endpoint (diffrax SaveAt(t1=True)) -- a
        # single point that is *not* a duplicate -- so every segment after
        # the first silently vanished from the returned trajectory: zs/us
        # collapsed to the front/back junction instead of spanning the
        # whole chain, even though the physics (the final state) was fine.
        def make_segment(zlo, zhi, seed):
            p = Propagator(1.55, fiber, Nmax=3, save_dir=save_dir)
            zs = np.linspace(zlo, zhi, 6)
            rng = np.random.default_rng(seed)
            p.neffs = 1.45 - 1e-4 * zs[:, None] - 1e-4 * np.arange(3)[None, :]
            C = rng.normal(size=(3, 3)) * 1e-4
            C = C - C.T
            p.cmats = zs[:, None, None] / zs[-1] * C[None, :, :]
            p.vs = np.zeros((len(zs), 3, 8))
            p.zs = zs
            p.make_interp_funcs()
            return p

        p1 = make_segment(0, 50, seed=0)
        p2 = make_segment(50, 100, seed=1)
        chain = ChainPropagator([p1, p2])
        u0 = np.zeros(3)
        u0[0] = 1.0
        zs, us, uf = chain.propagate(u0)

        # The true final z must be reached -- the bug got stuck reporting
        # the 50.0 front/back junction as the last point, since segment 2
        # contributed nothing. (On the jax backend propagate() reports only
        # ODE endpoints, never the start, so zs[0] == 50.0 here is expected
        # and not part of this regression -- only zs[-1] getting stuck at
        # the junction was the bug.)
        assert float(zs[-1]) == pytest.approx(100.0)
        assert len(zs) >= 2  # every segment contributed at least one point
        assert np.allclose(np.abs(us[-1]), np.abs(uf), atol=1e-8)

    def test_propagate_n_save_forwarded_per_segment(self, fiber, save_dir):
        # n_save is forwarded to each segment's own propagate() call, so a
        # 2-segment chain with n_save=5 should return a visibly dense
        # trajectory (not just the 2 junction/endpoint points jax gives by
        # default), reaching the true end of the chain either way.
        def make_segment(zlo, zhi, seed):
            p = Propagator(1.55, fiber, Nmax=3, save_dir=save_dir)
            zs = np.linspace(zlo, zhi, 6)
            rng = np.random.default_rng(seed)
            p.neffs = 1.45 - 1e-4 * zs[:, None] - 1e-4 * np.arange(3)[None, :]
            C = rng.normal(size=(3, 3)) * 1e-4
            C = C - C.T
            p.cmats = zs[:, None, None] / zs[-1] * C[None, :, :]
            p.vs = np.zeros((len(zs), 3, 8))
            p.zs = zs
            p.make_interp_funcs()
            return p

        p1 = make_segment(0, 50, seed=0)
        p2 = make_segment(50, 100, seed=1)
        chain = ChainPropagator([p1, p2])
        u0 = np.zeros(3)
        u0[0] = 1.0
        zs, us, uf = chain.propagate(u0, n_save=5)

        assert float(zs[0]) == pytest.approx(0.0)
        assert float(zs[-1]) == pytest.approx(100.0)
        assert len(zs) >= 9  # 5 + 5, minus the shared junction point
        assert np.allclose(np.abs(us[-1]), np.abs(uf), atol=1e-8)
