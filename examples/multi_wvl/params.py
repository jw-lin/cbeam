# Auto-split from the former single-file multi_wvl_pipeline.py.
"""Wavelength-dependent parameter scaling and the (wavelength-independent) lantern geometry."""
from __future__ import annotations
import numpy as np
from typing import Optional

from cbeam.waveguide import PhotonicLantern, hex_ring_positions


def build_lantern_geometry(base_params: dict) -> PhotonicLantern:
    """
    Build the PhotonicLantern geometry once. None of PhotonicLantern's
    constructor arguments (core positions, radii, indices, taper, mesh
    resolutions) depend on wavelength -- only the per-wavelength
    Propagator does. Rebuilding a fresh PhotonicLantern (and the
    underlying Gmsh/CAD geometry model) for every wavelength is pure
    repeated work, and if Gmsh's global model state isn't fully released
    between constructions, repeated builds can get progressively slower
    over a loop (a known pygmsh/Gmsh gotcha: the kernel's internal model
    keeps growing across successive builds in the same process). Build it
    once and share the same PL_N object across every wavelength's
    Propagator.
    """
    core_pos = hex_ring_positions(base_params["nrings"], base_params["rclad"] / 2.5)
    return PhotonicLantern(
        core_pos, base_params["rcores"], base_params["rclad"], base_params["rjack"],
        base_params["ncores"], base_params["nclad"], base_params["njack"], base_params["z_ex"],
        base_params["taper_factor"], base_params["core_res"], base_params["clad_res"],
        base_params["jack_res"],
    )



def sellmeier_n_silica(wavelength_um: float) -> float:
    """
    Refractive index of fused silica (Malitson 1965 Sellmeier fit).

    Kept as a utility for sizing the common-mode material-dispersion shift
    (see the CHROMATIC MODEL note above) -- it is *not* applied to the
    lantern indices by scale_params_to_wavelength(), by design.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometres (e.g. 0.800 for 800 nm).

    Returns
    -------
    n : float
        Refractive index at the given wavelength.
    """
    l2 = wavelength_um ** 2
    n2 = (1.0
          + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
          + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
          + 0.8974794 * l2 / (l2 - 9.896161  ** 2))
    return float(np.sqrt(n2))


def scale_params_to_wavelength(
    base_params: dict,
    wavelength_nm: float,
    ref_wavelength_nm: Optional[float] = None,
) -> dict:
    """
    Return a copy of *base_params* retuned for *wavelength_nm*.

    The lantern is a fixed physical object (see the CHROMATIC MODEL note
    above), so the refractive-index keys are left untouched -- geometry
    and dopant contrast are wavelength-independent. Only the two genuinely
    wavelength-dependent inputs are changed:

      * ``wl`` / ``wavelength_nm`` -- the vacuum wavelength that the
        per-wavelength FEM mode basis is solved/loaded at (this is what
        carries the k0 = 2π/λ and V ∝ 1/λ chromatic response).
      * ``pixel_scale_um`` -- Fraunhofer focal-plane sampling scales
        linearly with wavelength, Δξ ∝ λ, so
        pixel_scale_um(λ) = pixel_scale_um(λ₀) · λ/λ₀.

    Parameters
    ----------
    base_params : dict
        Output of get_simulation_parameters(). Must contain at least
        ``pixel_scale_um`` and ``wavelength_nm``.
    wavelength_nm : float
        Target wavelength in nanometres.
    ref_wavelength_nm : float or None
        Reference wavelength the pixel scale / base params are given at.
        Defaults to base_params["wavelength_nm"].

    Returns
    -------
    p_lambda : dict
        Copy of base_params with ``wl``, ``wavelength_nm`` and
        ``pixel_scale_um`` updated; every other key (including all
        refractive indices) is passed through unchanged.
    """
    if ref_wavelength_nm is None:
        ref_wavelength_nm = float(base_params["wavelength_nm"])

    p_lambda = dict(base_params)

    p_lambda["wl"]            = wavelength_nm / 1000.0
    p_lambda["wavelength_nm"] = wavelength_nm

    # Fraunhofer: focal-plane pixel pitch scales with wavelength.
    p_lambda["pixel_scale_um"] = (
        base_params["pixel_scale_um"] * (wavelength_nm / ref_wavelength_nm)
    )

    return p_lambda
