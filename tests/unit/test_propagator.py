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
