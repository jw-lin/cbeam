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
        rng=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a 2D spectral image from fiber fluxes.

        Parameters
        ----------
        modal_powers : ndarray (n_fibers,) or (n_wavelengths, n_fibers), or None
            Per-fiber power.  1D is treated as monochromatic and replicated
            across the wavelength grid.  Ignored when ``spectra`` is given.
        spectra : ndarray (n_wavelengths, n_fibers), optional
            Per-fiber flux density (per nm) at each wavelength; bin k
            contributes ``spectra[k] * delta_lambda`` ADU spread over the
            detector pixels its wavelength span covers.  ``n_wavelengths``
            must equal ``config.n_spectral_bins`` (the simulator's wavelength
            grid).  SpectralExtractor returns the same quantity.
        add_noise : bool
            Add dark current, Poisson shot noise and read noise.
        exposure_time : float
            Seconds, for the dark-current term.
        rng : None | int | np.random.Generator
            Noise source. None (default) uses numpy's global random state;
            an int seed or a Generator makes the noise reproducible.

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

        # Each bin's flux * delta_lambda is spread uniformly over the pixels
        # its wavelength span disperses onto (see
        # DispersionModel.bin_pixel_weights), so a continuous spectrum gives
        # a continuous trace whatever the binning.
        bin_width_px = self.delta_lambda * self.disp.pix_per_nm
        spread = self.disp.bin_pixel_weights() / bin_width_px        # (n_bins, n_x)
        traces = (np.maximum(spectra, 0.0) * self.delta_lambda).T @ spread
        for fiber_idx in range(n_fibers):
            self._add_trace(image, fiber_positions[fiber_idx], traces[fiber_idx])

        # seeing + instrument PSF
        image = gaussian_filter(
            image, sigma=(self.psf_sigma_spatial, self.psf_sigma_spectral))

        if add_noise:
            image = self._add_detector_noise(image, exposure_time, rng)

        return image, fiber_positions

    # ------------------------------------------------------------------
    def _add_trace(self, image, y_center, trace):
        """Add one fiber's dispersed trace (flux per detector column) at a
        fractional row, split linearly over the two surrounding rows so the
        sub-pixel position survives the later Gaussian convolution."""
        y0 = int(np.floor(y_center))
        fy = y_center - y0
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            yy = y0 + dy
            if wy > 0.0 and 0 <= yy < self.cfg.detector_height:
                image[yy, :] += wy * trace

    def _add_detector_noise(self, image, exposure_time, rng=None):
        """Dark current + Poisson shot noise + Gaussian read noise.

        ``image`` is in ADU and ``gain`` is electrons per ADU, so the noise is
        applied in electrons and converted back.
        """
        cfg = self.cfg
        electrons = np.maximum(image, 0.0) * cfg.gain
        electrons = electrons + cfg.dark_current * exposure_time
        rand = np.random if rng is None else np.random.default_rng(rng)
        electrons = rand.poisson(electrons).astype(np.float64)
        electrons = electrons + rand.normal(0.0, cfg.read_noise, electrons.shape)
        return np.maximum(electrons / cfg.gain, 0.0)
