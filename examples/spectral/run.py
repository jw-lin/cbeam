# Split out of the former single-file spectral_extraction_module.py.
"""Standalone demo: simulate 19 fiber spectra, disperse, extract, score."""
from __future__ import annotations
import numpy as np

from .config import SpectralConfig
from .simulate import SpectralImageSimulator
from .extract import SpectralExtractor
from .plots import visualize_spectral_image, visualize_extracted_spectra, visualize_all_spectra_grid


def example_spectral_extraction():
    """Demonstrate spectral extraction on simulated data."""
    print("=" * 70)
    print("SPECTRAL EXTRACTION EXAMPLE")
    print("=" * 70)

    config = SpectralConfig(
        lambda_min=700.0, lambda_max=900.0, n_spectral_bins=200,
        fiber_spacing_pixels=8.0, psf_fwhm_pixels=2.5,
    )
    simulator = SpectralImageSimulator(config)
    extractor = SpectralExtractor(config)

    # 19 fibers, each a Gaussian emission line at a different wavelength on a
    # per-fiber continuum.
    n_fibers = 19
    spectra_true = np.zeros((config.n_spectral_bins, n_fibers))
    for i in range(n_fibers):
        line_center = 750.0 + i * 5.0   # nm
        line_width = 2.0                # nm
        amplitude = 1000.0 + i * 50.0   # counts
        line_profile = amplitude * np.exp(
            -0.5 * ((simulator.wavelengths - line_center) / line_width) ** 2)
        continuum = 100.0 + i * 10.0
        spectra_true[:, i] = line_profile + continuum

    print("\n[1/3] Generating 2D spectral image...")
    image, fiber_positions = simulator.create_spectral_image(
        modal_powers=None, spectra=spectra_true, add_noise=True, exposure_time=1.0)

    print("[2/3] Visualizing spectral image...")
    visualize_spectral_image(image, fiber_positions, config)

    print("[3/3] Extracting 1D spectra...")
    spectra_extracted, variance = extractor.extract_spectra(
        image, fiber_positions, method="optimal")

    visualize_extracted_spectra(
        spectra_extracted, simulator.wavelengths, variance,
        fiber_indices=[0, 5, 10, 15, 18])
    visualize_all_spectra_grid(spectra_extracted, simulator.wavelengths)

    print("\n" + "=" * 70)
    print("EXTRACTION QUALITY METRICS")
    print("=" * 70)
    for i in [0, 5, 10, 15, 18]:
        residual = spectra_extracted[:, i] - spectra_true[:, i]
        rms = np.sqrt(np.mean(residual ** 2))
        finite_var = variance[:, i][np.isfinite(variance[:, i])]
        mean_var = np.mean(finite_var) if finite_var.size else np.inf
        snr = np.mean(spectra_true[:, i]) / np.sqrt(mean_var)
        print(f"Fiber {i:2d}:  RMS error = {rms:7.2f}  |  SNR = {snr:6.1f}")
    print("=" * 70)

    return image, spectra_extracted, spectra_true
