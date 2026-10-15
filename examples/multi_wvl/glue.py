# Auto-split from the former single-file multi_wvl_pipeline.py.
"""Glue helpers feeding multi-wavelength output into the ``spectral`` package."""
from __future__ import annotations
import numpy as np


def spectra_for_field(power_spectra: np.ndarray, field_idx: int) -> np.ndarray:
    """
    Slice out the (n_wavelengths, n_fibers) array for one field, in the
    shape expected by SpectralImageSimulator.create_spectral_image(spectra=...).
    """
    return power_spectra[:, field_idx, :]



def build_spectral_config_from_wavelength_grid(wl_nm: np.ndarray, **overrides):
    """
    Build a SpectralConfig whose lambda_min/lambda_max/n_spectral_bins match
    a wavelength grid produced by get_power_spectra(), so the spectral-
    extraction module's wavelength axis stays consistent with the
    physically propagated one (and with the dispersion fix discussed
    earlier -- SpectralExtractor must invert the same wavelength<->pixel
    relation SpectralImageSimulator used to build the image).
    """
    from spectral.config import SpectralConfig

    cfg_kwargs = dict(
        lambda_min=float(wl_nm.min()),
        lambda_max=float(wl_nm.max()),
        n_spectral_bins=len(wl_nm),
    )
    cfg_kwargs.update(overrides)
    return SpectralConfig(**cfg_kwargs)
