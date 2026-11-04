"""Core-centre detection, sub-pixel signal sampling and geometric alignment.

Pure processing -- this module imports no matplotlib.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter, map_coordinates
from scipy.optimize import linear_sum_assignment

from .constants import (
    backend, jax, jnp, N_SIGNALS, DEFAULT_SUBPIXEL_N, CALIB_GRID_RESOLUTION,
)


def batch_collect_subpixel_signals(images, coords):
    """Per-core signals for a whole batch of images in one call, on either
    backend (jit-compiled on jax).

    images : (n_images, n_rows, n_cols) intensity maps.
    coords : (2, n_cores, n, n) fractional (row, col) pixel coordinates of
        the n x n sampling window around each core.

    Returns (n_images, n_cores): the mean of each window, sampled with
    *bilinear* interpolation (jax's map_coordinates supports order <= 1) --
    so values differ slightly from collect_subpixel_signals, which samples
    the same windows with cubic interpolation.
    """
    return _batch_collect_subpixel_signals(images, coords)


def _batch_collect_subpixel_signals_np(images, coords):
    coords = np.asarray(coords)
    n_cores = coords.shape[1]
    flat = coords.reshape(2, -1)
    return np.stack([
        map_coordinates(np.asarray(img), flat, order=1, mode='constant', cval=0.0)
        .reshape(n_cores, -1).mean(axis=1)
        for img in images])


def _batch_collect_subpixel_signals_jax(images, coords):
    rows = coords[0]  # Shape: (N_SIGNALS, DEFAULT_SUBPIXEL_N, DEFAULT_SUBPIXEL_N)
    cols = coords[1]  # Shape: (N_SIGNALS, DEFAULT_SUBPIXEL_N, DEFAULT_SUBPIXEL_N)

    def extract_single_core_patch(img, r_coords, c_coords):
        c_pack = jnp.stack([r_coords, c_coords], axis=0)
        patch = jax.scipy.ndimage.map_coordinates(img, c_pack, order=1, mode='constant', cval=0.0)
        return jnp.mean(patch)

    collect_all_cores_single_frame = jax.vmap(
        lambda img: jax.vmap(lambda r, c: extract_single_core_patch(img, r, c))(rows, cols)
    )
    return collect_all_cores_single_frame(images)


if backend == 'jax':
    _batch_collect_subpixel_signals = jax.jit(_batch_collect_subpixel_signals_jax)
else:
    _batch_collect_subpixel_signals = _batch_collect_subpixel_signals_np


def _peak_centroids(intensity_2d, x0, y0, dx, dy, n_expected=N_SIGNALS,
                    filter_size=15, centroid_half_width=4,
                    max_attempts=15, thresh0=0.1, verbose=True):
    """Self-healing local-maxima finder + sub-pixel centroid refinement on a
    regular grid.

    Shared core of calibrate_subpixel_centers() (mesh profile -> griddata ->
    here) and detect_centers_from_grid() (already-gridded intensity -> here).

    ``(x0, y0)`` is the physical coordinate of grid index (0, 0) and
    ``(dx, dy)`` the grid spacing, so a peak at fractional index ``(row, col)``
    maps to ``(x0 + col*dx, y0 + row*dy)``.  Returns the centres sorted by
    ``(round(y, 2), round(x, 2))``.
    """
    thresh = thresh0
    peak_rows, peak_cols = np.array([], int), np.array([], int)
    for _ in range(max_attempts):
        is_peak = ((intensity_2d == maximum_filter(intensity_2d, size=filter_size))
                   & (intensity_2d > thresh * np.max(intensity_2d)))
        peak_rows, peak_cols = np.where(is_peak)
        if len(peak_rows) == n_expected:
            if verbose:
                print(f"[peaks] resolved {n_expected} cores at threshold {thresh:.3f}.")
            break
        thresh *= 0.75 if len(peak_rows) < n_expected else 1.25
    else:
        if verbose:
            print(f"[peaks] WARNING: converged on {len(peak_rows)} peaks, "
                  f"expected {n_expected}; using the closest configuration.")

    hw = centroid_half_width
    centers = []
    for r, c in zip(peak_rows, peak_cols):
        r0, r1 = max(0, r - hw), min(intensity_2d.shape[0], r + hw + 1)
        c0, c1 = max(0, c - hw), min(intensity_2d.shape[1], c + hw + 1)
        patch = intensity_2d[r0:r1, c0:c1]
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing='ij')
        s = np.sum(patch)
        if s > 0:
            sub_r = np.sum(rr * patch) / s
            sub_c = np.sum(cc * patch) / s
        else:
            sub_r, sub_c = r, c
        centers.append((x0 + sub_c * dx, y0 + sub_r * dy))

    return sorted(centers, key=lambda p: (np.round(p[1], 2), np.round(p[0], 2)))


def calibrate_subpixel_centers(total_modes_profile, mesh_final,
                               grid_resolution=CALIB_GRID_RESOLUTION):
    """Find core centres to sub-pixel accuracy: resample the mesh profile onto a
    ``grid_resolution`` square grid, then run the shared self-healing peak
    finder (_peak_centroids).

    Returns ``(centers, plot_x_out, plot_y_out, dx, dy)`` -- the grid axes and
    spacing are returned because callers (e.g. visualize_batch_hex_grid_signals)
    reuse them to sample signals at the same centres.
    """
    plot_x_out = np.linspace(mesh_final.points[:, 0].min(), mesh_final.points[:, 0].max(), grid_resolution)
    plot_y_out = np.linspace(mesh_final.points[:, 1].min(), mesh_final.points[:, 1].max(), grid_resolution)
    X_plot_out, Y_plot_out = np.meshgrid(plot_x_out, plot_y_out)
    dx = plot_x_out[1] - plot_x_out[0]
    dy = plot_y_out[1] - plot_y_out[0]

    total_modes_2d = griddata(
        (mesh_final.points[:, 0], mesh_final.points[:, 1]), total_modes_profile,
        (X_plot_out, Y_plot_out), method='cubic', fill_value=0.0,
    )

    centers = _peak_centroids(total_modes_2d, plot_x_out[0], plot_y_out[0], dx, dy,
                              n_expected=N_SIGNALS)
    return centers, plot_x_out, plot_y_out, dx, dy



def map_evaluated_to_ideal_geometry(core_centers, ideal_centers_list, max_iter=10):
    """Match detected core centres to the ideal port layout; returns
    ``ideal_permutation`` with ``ideal_permutation[k]`` = index in
    ``ideal_centers_list`` of detected core ``k`` (so
    ``standardized[ideal_permutation] = signals`` puts the signals in ideal
    order).

    Both sets are centred and scaled to the same RMS radius, then matched by
    minimum total distance (Hungarian assignment), alternating with a rigid
    Procrustes fit of the matched pairs to absorb a residual rotation, until
    the assignment stops changing. The order of either list does not matter.
    The two layouts are assumed to share their orientation up to that small
    residual rotation -- true for detected output cores against
    ``hex_ring_positions`` (the function the lantern geometry itself is built
    from); a hex grid fits itself equally well every 60 degrees, so a large
    rotation could not be recovered unambiguously anyway.
    """
    ideal_centers = np.array(ideal_centers_list, dtype=float)
    eval_centers = np.array(core_centers, dtype=float)

    if ideal_centers.shape != eval_centers.shape:
        raise ValueError(
            f"core_centers {eval_centers.shape} and ideal_centers "
            f"{ideal_centers.shape} must match: every detected core is "
            f"assigned to exactly one ideal port.")

    ideal_centered = ideal_centers - np.mean(ideal_centers, axis=0)
    eval_centered = eval_centers - np.mean(eval_centers, axis=0)

    scale_ideal = np.sqrt(np.mean(np.sum(ideal_centered**2, axis=1)))
    scale_eval = np.sqrt(np.mean(np.sum(eval_centered**2, axis=1)))
    eval_scaled = eval_centered * (scale_ideal / scale_eval)

    def assign(points):
        diff = points[:, np.newaxis, :] - ideal_centered[np.newaxis, :, :]
        cost = np.sqrt(np.sum(diff**2, axis=-1))
        _, perm = linear_sum_assignment(cost)
        return perm, cost[np.arange(len(perm)), perm]

    R = np.eye(2)
    ideal_permutation, dist = assign(eval_scaled)
    for _ in range(max_iter):
        H = np.dot(eval_scaled.T, ideal_centered[ideal_permutation])
        U, S, Vt = np.linalg.svd(H)
        # determinant correction: without it R can come out as a reflection
        # (improper rotation), which mirrors the detected grid before matching.
        d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
        R = np.dot(Vt.T, np.dot(np.diag([1.0, d]), U.T))
        eval_aligned = np.dot(eval_scaled, R.T)
        new_permutation, dist = assign(eval_aligned)
        if np.array_equal(new_permutation, ideal_permutation):
            break
        ideal_permutation = new_permutation

    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    print(f"[Geometric Fit] Detected/ideal scale ratio: {scale_eval / scale_ideal:.3f}")
    print(f"[Geometric Fit] Residual rotation: {angle:.3f} deg")
    print(f"[Geometric Fit] Residual matching RMS error: {np.sqrt(np.mean(dist**2)):.4e} "
          f"(in units of the ideal layout)\n")

    return ideal_permutation


def collect_subpixel_signals(image_2d, x_min, y_min, dx, dy, centers, n=DEFAULT_SUBPIXEL_N):
    """Extracts an n x n subimage centered on fractional continuous coordinates."""
    signals = np.zeros(len(centers))
    half_n = n // 2
    offsets = np.arange(-half_n, half_n + 1)
    
    for idx, (cx, cy) in enumerate(centers):
        f_col = (cx - x_min) / dx
        f_row = (cy - y_min) / dy
        
        sub_cols, sub_rows = np.meshgrid(f_col + offsets, f_row + offsets)
        coords = np.vstack((sub_rows.ravel(), sub_cols.ravel()))
        subimage_flat = map_coordinates(image_2d, coords, order=3, mode='constant', cval=0.0)
        signals[idx] = np.mean(subimage_flat)
        
    return signals


def detect_centers_from_grid(intensity_2d, X_plot, Y_plot, n_expected=N_SIGNALS):
    """Detect core centres from an already-gridded 2D intensity map.

    Thin wrapper over the shared self-healing peak finder (_peak_centroids):
    it derives the grid origin/spacing from the X_plot / Y_plot meshgrids and
    returns the sorted list of ``(x, y)`` centres.
    """
    x0, y0 = X_plot[0, 0], Y_plot[0, 0]
    dx = X_plot[0, 1] - X_plot[0, 0]
    dy = Y_plot[1, 0] - Y_plot[0, 0]
    return _peak_centroids(intensity_2d, x0, y0, dx, dy, n_expected=n_expected)

