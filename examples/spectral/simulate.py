# Split out of the former single-file spectral_extraction_module.py.
"""SpectralImageSimulator: build the 2D dispersed detector image. No matplotlib."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple

from .config import SpectralConfig
from .dispersion import DispersionModel


class SpectralImageSimulator:
    """
    Simulates the 2D spectral image formed by a dispersive spectrograph.

    Each photonic-lantern fiber output is dispersed along one detector axis,
    so one axis is spatial position (fiber ID) and the other is wavelength.
    """

    def __init__(self, config: SpectralConfig):
        self.cfg = config
        self.disp = DispersionModel(config)

        self.wavelengths = self.disp.wavelength_grid()
        self.delta_lambda = self.wavelengths[1] - self.wavelengths[0]

        # kept as a public attribute for backward compatibility
        self.dispersion_pix_per_nm = self.disp.pix_per_nm

        # PSF: convolution sigmas (the dispersion axis is optionally narrower)
        sigma = config.psf_fwhm_pixels / 2.355
        self.psf_sigma_spatial = sigma
        self.psf_sigma_spectral = sigma / config.psf_elongation

        print("SpectralImageSimulator initialized:")
        print(f"  Wavelength range: {config.lambda_min:.1f}-{config.lambda_max:.1f} nm")
        print(f"  Spectral bins: {config.n_spectral_bins}")
        print(f"  Dispersion: {self.dispersion_pix_per_nm:.3f} pix/nm")
        print(f"  Detector: {config.detector_width} x {config.detector_height} pixels")

    # ------------------------------------------------------------------
    def compute_fiber_positions(self, n_fibers: int = 19) -> np.ndarray:
        """Y-coordinate (row) of each fiber trace: n_fibers equally spaced,
        centred on the detector."""
        cfg = self.cfg
        total_extent = (n_fibers - 1) * cfg.fiber_spacing_pixels
        return cfg.detector_height / 2.0 + np.linspace(
            -total_extent / 2.0, total_extent / 2.0, n_fibers
        )

    def wavelength_to_pixel(self, wavelength):
        """nm -> detector pixel along the dispersion axis (delegates to DispersionModel)."""
        return self.disp.wavelength_to_pixel(wavelength)

    # ------------------------------------------------------------------
    def create_spectral_image(
        self,
        modal_powers: Optional[np.ndarray],
        spectra: Optional[np.ndarray] = None,
        add_noise: bool = True,
        exposure_time: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a 2D spectral image from fiber fluxes.

        Parameters
        ----------
        modal_powers : ndarray (n_fibers,) or (n_wavelengths, n_fibers), or None
            Per-fiber power.  1D is treated as monochromatic and replicated
            across the wavelength grid.  Ignored when ``spectra`` is given.
        spectra : ndarray (n_wavelengths, n_fibers), optional
            Per-fiber flux at each wavelength.  ``n_wavelengths`` must equal
            ``config.n_spectral_bins`` (the simulator's wavelength grid).
        add_noise : bool
            Add dark current, Poisson shot noise and read noise.
        exposure_time : float
            Seconds, for the dark-current term.

        Returns
        -------
        image : ndarray (detector_height, detector_width), detector ADU
        fiber_positions : ndarray (n_fibers,)
        """
        cfg = self.cfg

        if spectra is None:
            if modal_powers is None:
                raise ValueError("pass either modal_powers or spectra")
            modal_powers = np.asarray(modal_powers)
            if modal_powers.ndim == 1:
                spectra = np.tile(modal_powers, (cfg.n_spectral_bins, 1))
            elif modal_powers.ndim == 2:
                spectra = modal_powers
            else:
                raise ValueError(
                    f"modal_powers must be 1D or 2D, got shape {modal_powers.shape}")

        spectra = np.asarray(spectra)
        n_wavelengths, n_fibers = spectra.shape
        if n_wavelengths != cfg.n_spectral_bins:
            raise ValueError(
                f"spectra has {n_wavelengths} wavelength rows but the config "
                f"grid has {cfg.n_spectral_bins} bins; build the SpectralConfig "
                f"from the same wavelength grid (see "
                f"build_spectral_config_from_wavelength_grid).")

        fiber_positions = self.compute_fiber_positions(n_fibers)
        image = np.zeros((cfg.detector_height, cfg.detector_width), dtype=np.float64)

        x_centers = self.disp.wavelength_to_pixel(self.wavelengths)
        for fiber_idx in range(n_fibers):
            y_center = fiber_positions[fiber_idx]
            for wave_idx in range(n_wavelengths):
                flux = spectra[wave_idx, fiber_idx]
                if flux <= 0:
                    continue
                self._add_psf_spot(
                    image, y_center, x_centers[wave_idx], flux * self.delta_lambda)

        # seeing + instrument PSF
        image = gaussian_filter(
            image, sigma=(self.psf_sigma_spatial, self.psf_sigma_spectral))

        if add_noise:
            image = self._add_detector_noise(image, exposure_time)

        return image, fiber_positions

    # ------------------------------------------------------------------
    def _add_psf_spot(self, image, y_center, x_center, flux):
        """Deposit ``flux`` at a fractional (y, x), split bilinearly over the
        four surrounding pixels so sub-pixel dispersion positions survive the
        later Gaussian convolution."""
        cfg = self.cfg
        y0 = int(np.floor(y_center))
        x0 = int(np.floor(x_center))
        fy = y_center - y0
        fx = x_center - x0
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            yy = y0 + dy
            if wy == 0.0 or not (0 <= yy < cfg.detector_height):
                continue
            for dx, wx in ((0, 1.0 - fx), (1, fx)):
                xx = x0 + dx
                if wx == 0.0 or not (0 <= xx < cfg.detector_width):
                    continue
                image[yy, xx] += flux * wy * wx

    def _add_detector_noise(self, image, exposure_time):
        """Dark current + Poisson shot noise + Gaussian read noise.

        ``image`` is in ADU and ``gain`` is electrons per ADU, so the noise is
        applied in electrons and converted back.
        """
        cfg = self.cfg
        electrons = np.maximum(image, 0.0) * cfg.gain
        electrons = electrons + cfg.dark_current * exposure_time
        electrons = np.random.poisson(electrons).astype(np.float64)
        electrons = electrons + np.random.normal(0.0, cfg.read_noise, electrons.shape)
        return np.maximum(electrons / cfg.gain, 0.0)
