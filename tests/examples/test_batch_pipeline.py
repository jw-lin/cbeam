"""Smoke tests for ``examples/batch_pipeline`` (the batched-propagation examples).

The fast tests cover the influence-function loading (with and without
specula) and the pupil -> FFT -> mesh field generation, on a small synthetic
pupil.  The one ``slow`` test runs the whole pipeline -- characterization,
field generation, batched propagation, output reconstruction and per-core
signal extraction -- on a shortened 19-port lantern, as ``test_pl19.py``
does for the plain ``ChainPropagator``.
"""

import pathlib
import sys

import numpy as np
import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import batch_pipeline as bp  # noqa: E402
from batch_pipeline.engine import IncidentFieldGenerator  # noqa: E402

BUNDLED_IFUNC = EXAMPLES / "ifunc" / "ANDES_400pix_all_modes.fits"


def _synthetic_ifunc(npix=64, radius=None, offset=(0, 0)):
    """Circular pupil with tip / tilt / focus modes, as a ``SimpleIFunc``."""
    radius = npix / 2 - 2 if radius is None else radius
    yy, xx = np.mgrid[:npix, :npix].astype(float)
    cy, cx = (npix - 1) / 2 + offset[0], (npix - 1) / 2 + offset[1]
    y, x = (yy - cy) / radius, (xx - cx) / radius
    mask = (x ** 2 + y ** 2) <= 1.0
    modes = np.stack([x[mask], y[mask], 2 * (x[mask] ** 2 + y[mask] ** 2) - 1])
    return bp.SimpleIFunc(modes, mask)


def _field_params(npix):
    p = bp.get_simulation_parameters(3, 0.8)
    # keep the padded grid's physical extent the same as for the 400 px pupil
    p["pixel_scale_um"] *= 400 / npix
    return p


# --------------------------------------------------------------------------- #
# influence-function loading
# --------------------------------------------------------------------------- #
def test_simple_ifunc_rejects_mismatched_shapes():
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    bp.SimpleIFunc(np.zeros((3, 16)), mask)
    with pytest.raises(ValueError):
        bp.SimpleIFunc(np.zeros((3, 15)), mask)


def test_center_ifunc_mask_moves_only_the_mask():
    ifunc = _synthetic_ifunc(offset=(3, -2))
    mask_before = ifunc.mask_inf_func.copy()
    modes_before = ifunc.influence_function.copy()
    shift_y, shift_x = bp.center_ifunc_mask(ifunc)
    assert (shift_y, shift_x) != (0, 0)
    np.testing.assert_array_equal(
        ifunc.mask_inf_func, np.roll(mask_before, (shift_y, shift_x), axis=(0, 1)))
    rows, cols = np.nonzero(ifunc.mask_inf_func)
    assert abs(rows.mean() - 32) < 1 and abs(cols.mean() - 32) < 1
    assert ifunc.mask_inf_func.dtype == np.float32
    np.testing.assert_array_equal(ifunc.influence_function, modes_before)


def test_center_ifunc_mask_refuses_to_wrap():
    mask = np.zeros((16, 16), bool)
    mask[0:2, 0:2] = True          # centroid far off-centre, touching the edge
    mask[15, 15] = True            # ... and a pixel on the opposite edge
    ifunc = bp.SimpleIFunc(np.zeros((1, 5)), mask)
    with pytest.raises(ValueError):
        bp.center_ifunc_mask(ifunc)


def test_bundled_ifunc_without_specula():
    pytest.importorskip("astropy")
    ifunc = bp.load_ifunc(BUNDLED_IFUNC, use_specula=False)
    assert isinstance(ifunc, bp.SimpleIFunc)
    assert ifunc.mask_inf_func.shape == (400, 400)
    assert ifunc.influence_function.shape == (10, 109096)
    assert ifunc.influence_function.dtype == np.float64
    assert ifunc.mask_inf_func.dtype == np.float32


def test_bundled_ifunc_specula_and_numpy_bit_identical():
    """What the pipeline sees must not depend on whether specula is installed."""
    pytest.importorskip("astropy")
    pytest.importorskip("specula")
    p = bp.get_simulation_parameters(3, 0.8)
    gens = []
    for use_specula in (True, False):
        ifunc = bp.load_ifunc(BUNDLED_IFUNC, use_specula=use_specula)
        bp.center_ifunc_mask(ifunc)
        gens.append(IncidentFieldGenerator(p, ifunc, xp=np))
    sp, nps = gens
    np.testing.assert_array_equal(sp.mask_np, nps.mask_np)
    assert sp.ifunc_matrix.dtype == nps.ifunc_matrix.dtype
    np.testing.assert_array_equal(sp.ifunc_matrix, nps.ifunc_matrix)


# --------------------------------------------------------------------------- #
# field generation
# --------------------------------------------------------------------------- #
def _mesh_points(n=500, r=20.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-r, r, size=(n, 2))
    return np.vstack([[0.0, 0.0], pts])          # point 0: the optical axis


def test_flat_pupil_focuses_on_axis():
    """A zero-aberration pupil's FFT peaks on axis, with value = pupil area."""
    ifunc = _synthetic_ifunc()
    p = _field_params(64)
    fg = IncidentFieldGenerator(p, ifunc, xp=np)
    fg.precompute_interpolation_weights(_mesh_points())

    E_pupil = fg.generate_field_profiles_batch(np.zeros((1, fg.num_modes)))
    E_focal = fg.apply_ef_to_lantern(E_pupil)
    E_mesh = fg.resample_to_mesh(E_focal)

    n_pix = int(fg.mask_np.sum())
    assert E_focal.shape == (1, 256, 256)
    assert abs(E_focal[0, 128, 128] - n_pix) < 1e-9 * n_pix
    assert np.argmax(np.abs(E_mesh[0])) == 0
    assert abs(E_mesh[0, 0] - n_pix) < 1e-9 * n_pix


def test_field_generation_numpy_vs_jax(backend):
    if backend != "jax":
        pytest.skip("needs CBEAM_BACKEND=jax")
    import jax.numpy as jnp

    ifunc = _synthetic_ifunc()
    p = _field_params(64)
    pts = _mesh_points()
    coeffs = np.random.default_rng(1).uniform(-300, 300, size=(4, 3))

    fg_np = IncidentFieldGenerator(p, ifunc, xp=np)
    fg_np.precompute_interpolation_weights(pts)
    ref = fg_np.resample_to_mesh(
        fg_np.apply_ef_to_lantern(fg_np.generate_field_profiles_batch(coeffs)))

    fg_jx = IncidentFieldGenerator(p, ifunc, xp=jnp)
    fg_jx.precompute_interpolation_weights(jnp.asarray(pts))
    out = fg_jx.resample_to_mesh(
        fg_jx.apply_ef_to_lantern_jax(fg_jx.generate_field_profiles_batch(jnp.asarray(coeffs))))

    scale = np.abs(ref).max()
    np.testing.assert_allclose(np.asarray(out), ref, rtol=0, atol=1e-9 * scale)


# --------------------------------------------------------------------------- #
# configs, core matching, signal extraction
# --------------------------------------------------------------------------- #
def test_random_configs_rng():
    a, _ = bp.create_random_aberration_configs(5, 3, -1, 1, rng=7)
    b, _ = bp.create_random_aberration_configs(5, 3, -1, 1, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    s1, _ = bp.create_sparse_aberration_configs_mono(5, 3, -1, 1, rng=7)
    s2, _ = bp.create_sparse_aberration_configs_mono(5, 3, -1, 1, rng=7)
    np.testing.assert_array_equal(s1, s2)
    # rng=None keeps drawing from numpy's global state
    np.random.seed(3)
    c, _ = bp.create_random_aberration_configs(5, 3, -1, 1)
    np.random.seed(3)
    np.testing.assert_array_equal(c, np.random.uniform(-1, 1, (5, 3)))


def test_core_matching_does_not_depend_on_order():
    from cbeam.waveguide import hex_ring_positions

    ideal = np.asarray(hex_ring_positions(3, 1.0))
    rng = np.random.default_rng(0)
    a = np.radians(2.0)                              # small residual rotation
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    truth = rng.permutation(len(ideal))              # detected k <-> ideal truth[k]
    detected = 43.2 * ideal[truth] @ R.T + [5.0, -3.0]
    detected += rng.normal(scale=0.5, size=detected.shape)

    np.testing.assert_array_equal(bp.map_evaluated_to_ideal_geometry(detected, ideal), truth)
    # shuffling the ideal list too only relabels the answer
    shuffle = rng.permutation(len(ideal))
    perm = bp.map_evaluated_to_ideal_geometry(detected, ideal[shuffle])
    np.testing.assert_array_equal(shuffle[perm], truth)


def test_batch_collect_subpixel_signals():
    from scipy.ndimage import map_coordinates

    rng = np.random.default_rng(0)
    images = rng.random((3, 40, 40))
    coords = rng.uniform(5, 35, size=(2, 4, 5, 5))
    ref = np.array([[map_coordinates(img, coords[:, k].reshape(2, -1), order=1).mean()
                     for k in range(4)] for img in images])
    out = np.asarray(bp.batch_collect_subpixel_signals(images, coords))
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out, ref, rtol=1e-12)


# --------------------------------------------------------------------------- #
# whole pipeline
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.integration
def test_batch_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)     # characterization is saved under ./data

    p = bp.get_simulation_parameters(3, 0.8)
    # shortened device and coarser mesh, as in test_pl19.py
    p.update(L1=4000, L2=4000, z_ex=8000, core_res=12, clad_res=40, jack_res=24)
    prop12 = bp.build_and_characterize_lantern(p)

    ifunc = _synthetic_ifunc(npix=400)
    pipeline = bp.BatchPropagationPipeline(prop12, p, ifunc)

    coeffs = np.zeros((4, 3))
    coeffs[1, 0] = 150.0                # tip
    coeffs[2, 1] = -150.0               # tilt
    coeffs[3, 2] = 100.0                # focus
    u0, eff = pipeline.generate_batch_modal_coefficients(
        coeffs, return_coupling_efficiency=True)
    u0, eff = np.asarray(u0), np.asarray(eff)
    assert u0.shape == (4, prop12.Nmax)
    np.testing.assert_allclose(np.linalg.norm(u0, axis=1), 1.0, atol=1e-12)
    # (unnormalised overlap norms, so only their sign/finiteness is checked)
    assert np.all(np.isfinite(eff) & (eff > 0))

    uf, zs, us = pipeline.propagate_batch(u0)
    uf = np.asarray(uf)
    assert float(zs[-1]) == pytest.approx(p["z_ex"])
    # power is conserved, apart from what sits in the skipped mode at launch
    kept = [j for j in range(prop12.Nmax) if j not in prop12.skipped_modes]
    p_in = np.sum(np.abs(u0[:, kept]) ** 2, axis=1)
    np.testing.assert_allclose(np.sum(np.abs(uf) ** 2, axis=1), p_in, atol=0.05)

    # one batched call == the same fields propagated one at a time
    for i in (0, 3):
        _, _, uf_i = prop12.propagate(u0[i])
        np.testing.assert_allclose(uf[i], np.asarray(uf_i), atol=1e-6)

    E = pipeline.reconstruct_batch_output_fields(uf)
    grid, X, Y = pipeline.interpolate_output_to_grid(E, grid_resolution=120)
    grid = np.asarray(grid)
    assert grid.shape == (4, 120, 120)
    assert np.all(np.isfinite(grid)) and np.all(grid >= 0)

    # on this short device only a few of the 19 output cores light up
    centers = bp.detect_centers_from_grid(grid.sum(axis=0), X, Y)
    assert 1 <= len(centers) <= bp.N_SIGNALS
    x0, y0 = X[0, 0], Y[0, 0]
    dx, dy = X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0]
    signals = np.array([bp.collect_subpixel_signals(g, x0, y0, dx, dy, centers)
                        for g in grid])
    assert signals.shape == (4, len(centers))
    assert np.all(signals >= 0) and np.all(signals.sum(axis=1) > 0)
