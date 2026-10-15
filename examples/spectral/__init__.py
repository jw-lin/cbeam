# Split out of the former single-file spectral_extraction_module.py.
"""Dispersive imaging of photonic-lantern outputs + optimal spectral extraction.

Pipeline stages:
  1. Propagate through the photonic lantern (batch_pipeline / multi_wvl)
  2. Project fiber outputs onto the slit plane
  3. Apply wavelength-dependent dispersion               -> DispersionModel
  4. Build the 2D spectral image (spatial x wavelength)  -> SpectralImageSimulator
  5. Extract 1D spectra by optimal extraction            -> SpectralExtractor

``simulate``/``extract``/``config``/``dispersion`` are free of matplotlib;
``plots`` is the only module that imports it.
"""
from __future__ import annotations

from .config import SpectralConfig
from .dispersion import DispersionModel
from .simulate import SpectralImageSimulator
from .extract import SpectralExtractor
from .plots import (
    visualize_spectral_image,
    visualize_extracted_spectra,
    visualize_all_spectra_grid,
)
from .run import example_spectral_extraction

__all__ = [
    "SpectralConfig",
    "DispersionModel",
    "SpectralImageSimulator",
    "SpectralExtractor",
    "visualize_spectral_image",
    "visualize_extracted_spectra",
    "visualize_all_spectra_grid",
    "example_spectral_extraction",
]
