# Split out of the former single-file spectral_extraction_module.py.
"""SpectralExtractor: optimal / aperture extraction of 1D spectra. No matplotlib."""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

from .config import SpectralConfig
from .dispersion import DispersionModel


class SpectralExtractor:
    """
    Extract 1D spectra from a 2D spectral image.

    Optimal extraction follows Horne (1986, PASP 98, 609): a normalised
    spatial profile per fiber, variance-weighted collapse along the spatial
    axis, then resampling onto the wavelength grid through the *same*
    dispersion relation the simulator used (DispersionModel).
    """

    def __init__(self, config: SpectralConfig):
        self.cfg = config
        self.disp = DispersionModel(config)
        # spatial profiles, cached together with the fiber_positions they were
        # built for so a later call with different positions rebuilds them
        self.profiles: Optional[List[np.ndarray]] = None
        self._profiles_key: Optional[bytes] = None

    # ------------------------------------------------------------------
    def extract_spectra(
        self,
        image: np.ndarray,
        fiber_positions: np.ndarray,
        method: str = "optimal",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        spectra  : ndarray (n_spectral_bins, n_fibers)
        variance : ndarray (n_spectral_bins, n_fibers)  (np.inf where a bin is
                   outside the illuminated part of that fiber's trace)
        """
        cfg = self.cfg
        fiber_positions = np.asarray(fiber_positions, dtype=float)
        n_fibers = len(fiber_positions)

        spectra = np.zeros((cfg.n_spectral_bins, n_fibers))
        variance = np.zeros((cfg.n_spectral_bins, n_fibers))

        variance_map = self._estimate_variance_map(image)
        self._ensure_profiles(fiber_positions)

        for fiber_idx in range(n_fibers):
            y_min, y_max = self._get_extraction_aperture(fiber_positions[fiber_idx])
            if method == "optimal":
                spec, var = self._optimal_extraction(
                    image, variance_map, y_min, y_max, fiber_idx)
            else:
                spec, var = self._aperture_extraction(
                    image, variance_map, y_min, y_max)
            spectra[:, fiber_idx] = spec
            variance[:, fiber_idx] = var

        return spectra, variance

    # ------------------------------------------------------------------
    def pixel_to_wavelength(self, pixel):
        """Detector pixel (dispersion axis) -> nm.  Inverse of what the
        simulator used; delegates to DispersionModel."""
        return self.disp.pixel_to_wavelength(pixel)

    # -- internals ---------------------------------------------------
    def _estimate_variance_map(self, image: np.ndarray) -> np.ndarray:
        """Per-pixel variance in ADU**2 (Poisson + read noise)."""
        cfg = self.cfg
        signal_e = np.maximum(image, 0.0) * cfg.gain
        var_e = signal_e + cfg.read_noise ** 2
        return var_e / cfg.gain ** 2

    def _ensure_profiles(self, fiber_positions: np.ndarray) -> None:
        key = fiber_positions.tobytes()
        if self.profiles is not None and self._profiles_key == key:
            return
        self.profiles = self._build_spatial_profiles(fiber_positions)
        self._profiles_key = key

    def _build_spatial_profiles(self, fiber_positions: np.ndarray) -> List[np.ndarray]:
        """Normalised Gaussian spatial profile per fiber (sum = 1), clipped to
        the same aperture the extraction uses."""
        cfg = self.cfg
        sigma = cfg.psf_fwhm_pixels / 2.355
        profiles = []
        for y_center in fiber_positions:
            y_min, y_max = self._get_extraction_aperture(y_center)
            y_coords = np.arange(y_min, y_max)
            profile = np.exp(-0.5 * ((y_coords - y_center) / sigma) ** 2)
            profile /= profile.sum()
            profiles.append(profile)
        return profiles

    def _get_extraction_aperture(self, y_center: float) -> Tuple[int, int]:
        cfg = self.cfg
        sigma = cfg.psf_fwhm_pixels / 2.355
        half_width = int(np.ceil(cfg.extraction_aperture_sigma * sigma))
        y_pix = int(np.round(y_center))
        return (max(0, y_pix - half_width),
                min(cfg.detector_height, y_pix + half_width + 1))

    def _resample_to_wavelength_grid(self, spectrum_x, variance_x):
        """Resample a per-column (detector x) spectrum/variance onto the config
        wavelength grid, using the simulator's dispersion relation."""
        x_coords = np.arange(spectrum_x.shape[0])
        x_pixels = self.disp.wavelength_to_pixel(self.disp.wavelength_grid())
        spec = np.interp(x_pixels, x_coords, spectrum_x, left=0.0, right=0.0)
        var = np.interp(x_pixels, x_coords, variance_x, left=np.inf, right=np.inf)
        return spec, var

    def _aperture_extraction(self, image, variance_map, y_min, y_max):
        """Plain sum of pixels in the spatial aperture."""
        spectrum = np.sum(image[y_min:y_max, :], axis=0)
        variance = np.sum(variance_map[y_min:y_max, :], axis=0)
        return self._resample_to_wavelength_grid(spectrum, variance)

    def _optimal_extraction(self, image, variance_map, y_min, y_max, fiber_idx):
        """
        Horne 1986: flux = sum(P*D/V) / sum(P**2/V), var = 1 / sum(P**2/V),
        with P the normalised spatial profile, D the data, V the variance.
        Columns where the trace is not illuminated (denominator ~ 0) are
        returned as flux 0, variance inf rather than blown up by a fudge term.
        """
        sub_image = image[y_min:y_max, :]
        sub_variance = variance_map[y_min:y_max, :]
        profile = self.profiles[fiber_idx][:, np.newaxis]      # (n_spatial, 1)

        weights = profile ** 2 / sub_variance
        numerator = np.sum(profile * sub_image * weights, axis=0)
        denominator = np.sum(profile ** 2 * weights, axis=0)

        # denominator is > 0 wherever the (strictly positive) profile overlaps
        # finite-variance pixels; guard the degenerate case explicitly instead
        # of the old `+ 1e-10` fudge, which silently produced huge flux/variance
        # spikes in any column where the trace carried no signal.
        valid = np.isfinite(denominator) & (denominator > 0)
        safe_denom = np.where(valid, denominator, 1.0)
        spectrum = np.where(valid, numerator / safe_denom, 0.0)
        variance = np.where(valid, 1.0 / safe_denom, np.inf)

        return self._resample_to_wavelength_grid(spectrum, variance)
