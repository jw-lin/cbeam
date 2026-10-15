# Split out of the former single-file spectral_extraction_module.py.
"""SpectralConfig: every tunable parameter of the simulate/extract pipeline."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SpectralConfig:
    """Configuration parameters for spectral extraction."""

    # Wavelength range
    lambda_min: float = 700.0      # nm
    lambda_max: float = 900.0      # nm
    n_spectral_bins: int = 200     # spectral resolution elements

    # Slit geometry
    fiber_spacing_pixels: float = 8.0        # spatial separation on detector (pixels)
    fiber_core_diameter_pixels: float = 3.0  # projected fiber size

    # Dispersion
    dispersion_axis: int = 1              # 0=vertical, 1=horizontal
    dispersion_mm_per_nm: float = 0.05   # detector plate scale
    pixel_size_um: float = 15.0          # detector pixel size

    # PSF model
    psf_fwhm_pixels: float = 2.5   # seeing-limited PSF
    psf_elongation: float = 1.0    # anamorphic factor (1.0 = circular);
    #                                only the dispersion-axis sigma is divided by it

    # Detector
    detector_width: int = 2048     # pixels
    detector_height: int = 512     # pixels
    read_noise: float = 3.0        # electrons RMS
    dark_current: float = 0.1      # electrons/pixel/sec
    gain: float = 1.0              # electrons per ADU

    # Extraction
    extraction_aperture_sigma: float = 3.0  # aperture = +/-N sigma
    background_width_pixels: int = 5         # width of background regions
    optimal_extraction: bool = True          # use variance weighting
