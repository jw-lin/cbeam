"""Simulation-parameter builder and aberration-configuration generators."""
from __future__ import annotations
import os
import numpy as np

from cbeam.waveguide import hex_ring_positions

from .constants import (
    L1, L2, N_RINGS, DEFAULT_WAVELENGTH_UM,
)

# Influence-function FITS (the ANDES DM modal basis), bundled under
# examples/ifunc/.  Point CBEAM_IFUNC_FILE at another specula-format FITS
# file to use a different one.
DEFAULT_IFUNC_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'ifunc', 'ANDES_400pix_all_modes.fits',
)


def get_simulation_parameters(nrings=N_RINGS, wavelength_um=DEFAULT_WAVELENGTH_UM):
    """Return a dict of all setup constants and derived parameters."""

    wavelength_nm = wavelength_um * 1000.0
    scaling_factor = wavelength_um / DEFAULT_WAVELENGTH_UM
    # scaling_factor = 1.0

    params = {
        "nrings":        nrings,
        "wl":            wavelength_um,
        "wavelength_nm": wavelength_nm,
        "taper_factor":  12.,
        "rclad":         9.0,
        "rjack":         27,
        "L1":            L1,
        "L2":            L2,
        "z_ex":          L1 + L2,
        "nclad":         1.444,
        "pad_factor":    4,
        "core_res":      16,
        "clad_res":      60,
        "jack_res":      30,
        # "pixel_scale_um" (μm/px in the padded FFT grid) is derived below.
        "ifunc_file":    os.environ.get("CBEAM_IFUNC_FILE", DEFAULT_IFUNC_FILE),
    }
    params["rcore"]  = 1.8 / params["taper_factor"]
    params["ncore"]  = params["nclad"] + 8.8e-3
    params["njack"]  = params["nclad"] - 5.5e-3
    params["output_positions"] = hex_ring_positions(params["nrings"], params["rclad"] / 2.5)
    params["n_output_positions"] = len(params["output_positions"])
    params["rcores"] = [params["rcore"]] * params["n_output_positions"]
    params["ncores"] = [params["ncore"]] * params["n_output_positions"]
    
    # Focal-plane (post-FFT) pixel scale.
    #
    # The pupil field is sampled on grid_size px and zero-padded to
    # grid_size * pad_factor before fft2, so the diffraction pattern is
    # sampled at exactly `pad_factor` px per (lambda/D).  The Airy first-null
    # radius is therefore 1.22 * pad_factor px -- it does NOT scale with
    # grid_size (the earlier `* grid_size / 2` term shrank pixel_scale_um by
    # ~200x, so the padded grid only spanned +-5 um and ~2/3 of the +-27 um
    # lantern mesh fell outside it -> "mesh points map outside FFT grid").
    #
    # pixel_scale_um is then fixed by requiring the Airy radius to equal
    # psf_fill_factor * rclad in physical units.
    psf_fill_factor = 0.30
    r_airy_px = 1.22 * params["pad_factor"]
    params["pixel_scale_um"] = psf_fill_factor * params["rclad"] / r_airy_px

    return params


def create_sparse_aberration_configs_mono(n, m, minv, maxv, rng=None):
    """
    Generate *n* random aberration configurations where each configuration
    has only one active mode (value) that varies, while others remain zero.

    Args:
        n    (int): Number of configurations.
        m    (int): Number of modes available (indices 0 to m-1).
        minv (float): Minimum amplitude (nm).
        maxv (float): Maximum amplitude (nm).
        rng  (None | int | np.random.Generator): random source. None
                     (default) draws from numpy's global random state, so
                     np.random.seed() controls it; an int seed or a
                     Generator draws from that generator instead.

    Returns:
        coeff_matrix : np.ndarray of shape (n, m), dtype float64.
        labels       : list[str] of length n, e.g. "mode 3 | 150.0 nm"
                       (same convention as create_ramp_aberration_configs).
    """
    configs = np.zeros((n, m), dtype=np.float64)
    if rng is None:
        random_mode_indices = np.random.randint(0, m, size=n)
        amplitudes = np.random.uniform(minv, maxv, size=n)
    else:
        rng = np.random.default_rng(rng)
        random_mode_indices = rng.integers(0, m, size=n)
        amplitudes = rng.uniform(minv, maxv, size=n)
    configs[np.arange(n), random_mode_indices] = amplitudes

    labels = [f"mode {int(i)} | {a:.1f} nm"
              for i, a in zip(random_mode_indices, amplitudes)]
    return configs, labels


def create_random_aberration_configs(n, m, minv, maxv, rng=None):
    """
    Generate *n* random aberration configurations each with *m* active modes.

    Args:
        n    (int):   Number of configurations.
        m    (int):   Number of active modes per config (modes 0 … m-1).
        minv (float): Minimum amplitude (nm).
        maxv (float): Maximum amplitude (nm).
        rng  (None | int | np.random.Generator): random source. None
                     (default) draws from numpy's global random state, so
                     np.random.seed() controls it; an int seed or a
                     Generator draws from that generator instead.

    Returns:
        coeff_matrix : np.ndarray of shape (n, m), dtype float64.
        labels       : list[str] of length n, "random #k | rms <v> nm"
                       (no single active mode, so the label reports the
                       per-config RMS amplitude instead).
    """
    if rng is None:
        configs = np.random.uniform(minv, maxv, (n, m)).astype(np.float64)
    else:
        configs = np.random.default_rng(rng).uniform(minv, maxv, (n, m))
    labels = [f"random #{k} | rms {np.sqrt(np.mean(row ** 2)):.0f} nm"
              for k, row in enumerate(configs)]
    return configs, labels


def create_ramp_aberration_configs(modes, n_steps, minv, maxv):
    """
    Generate aberration configurations that ramp each mode individually.

    For every mode index in *modes* a sequence of *n_steps* configurations is
    produced in which that mode's amplitude is varied linearly from *minv* to
    *maxv* while all other modes are kept at zero.  The resulting array has
    ``len(modes) * n_steps`` rows.

    Args:
        modes   (list[int]): mode indices to ramp, e.g. [0, 1, 2].
        n_steps (int):       number of amplitude steps per mode (≥ 2).
        minv    (float):     starting amplitude value (nm).
        maxv    (float):     ending amplitude value (nm).

    Returns:
        coeff_matrix : np.ndarray, shape (len(modes) * n_steps, max(modes)+1),
                       dtype float64.
                       Each row is one configuration; the column index equals
                       the mode index.
        labels       : list[str] of length len(modes) * n_steps, one human-
                       readable label per configuration, e.g.
                       "mode 2 | 150.0 nm".

    Example:
        # Ramp modes 0, 3, 5 over 10 steps from 0 to 500 nm
        coeff, labels = create_ramp_aberration_configs([0, 3, 5], 10, 0, 500)
        u0_batch = pipeline.generate_batch_modal_coefficients(coeff)
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2 to define a ramp.")
    if len(modes) == 0:
        raise ValueError("modes list must not be empty.")

    amplitudes  = np.linspace(minv, maxv, n_steps)   # shape (n_steps,)
    n_cols      = max(modes) + 1
    n_configs   = len(modes) * n_steps

    coeff_matrix = np.zeros((n_configs, n_cols), dtype=np.float64)
    labels       = []

    for row, (mode_idx, amp) in enumerate(
            (m, a) for m in modes for a in amplitudes):
        coeff_matrix[row, mode_idx] = amp
        labels.append(f"mode {mode_idx} | {amp:.1f} nm")

    return coeff_matrix, labels

