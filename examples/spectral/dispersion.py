"""DispersionModel: the one wavelength <-> detector-pixel mapping.

Both halves of the pipeline need this relation and they must agree exactly:
``SpectralImageSimulator`` uses it to lay each fiber trace down on the
detector, and ``SpectralExtractor`` must invert *that same* relation when it
integrates the along-trace cut back onto the wavelength grid -- keeping both
directions in one place keeps them consistent.
"""
from __future__ import annotations
import numpy as np

from .config import SpectralConfig


class DispersionModel:
    """Linear dispersion: wavelength maps to a pixel coordinate along the
    dispersion axis, with the band centre pinned to the detector centre."""

    def __init__(self, config: SpectralConfig):
        if config.dispersion_axis != 1:
            raise ValueError(
                "only dispersion_axis=1 (wavelength along the detector rows, "
                "i.e. horizontal) is implemented; got "
                f"dispersion_axis={config.dispersion_axis}")
        self.cfg = config
        # pixels per nm along the dispersion axis
        self.pix_per_nm = (
            config.dispersion_mm_per_nm * 1000.0 / config.pixel_size_um
        )

    # -- geometry -------------------------------------------------------
    @property
    def axis_length(self) -> int:
        """Detector extent (px) along the dispersion axis."""
        cfg = self.cfg
        return cfg.detector_width if cfg.dispersion_axis == 1 else cfg.detector_height

    @property
    def centre_offset(self) -> float:
        """Pixel coordinate that ``(lambda_min + lambda_max) / 2`` lands on --
        i.e. the dispersed band is centred on the detector."""
        span_px = (self.cfg.lambda_max - self.cfg.lambda_min) * self.pix_per_nm
        return self.axis_length / 2.0 - span_px / 2.0

    # -- the mapping, both ways ---------------------------------------
    def wavelength_to_pixel(self, wavelength):
        """nm -> pixel coordinate along the dispersion axis (float or array)."""
        return (np.asarray(wavelength, dtype=float) - self.cfg.lambda_min) \
            * self.pix_per_nm + self.centre_offset

    def pixel_to_wavelength(self, pixel):
        """pixel coordinate along the dispersion axis -> nm (float or array)."""
        return self.cfg.lambda_min + (
            np.asarray(pixel, dtype=float) - self.centre_offset
        ) / self.pix_per_nm

    def wavelength_grid(self) -> np.ndarray:
        """The ``n_spectral_bins`` wavelength samples spanning the band."""
        return np.linspace(
            self.cfg.lambda_min, self.cfg.lambda_max, self.cfg.n_spectral_bins
        )

    def bin_width_nm(self) -> float:
        """Width of one spectral bin: the spacing of wavelength_grid()."""
        cfg = self.cfg
        return (cfg.lambda_max - cfg.lambda_min) / (cfg.n_spectral_bins - 1)

    def bin_pixel_weights(self) -> np.ndarray:
        """(n_spectral_bins, axis_length) overlap, in pixels, of each spectral
        bin's span along the dispersion axis with each detector pixel.

        Bin k covers ``wavelength_to_pixel(lambda_k) +- bin_width/2`` and pixel
        j covers ``[j - 0.5, j + 0.5]`` (pixel centres at integer
        coordinates), so the bins tile the dispersed band without gaps or
        overlaps and every row sums to the bin's width in pixels (less any
        part falling off the detector).  The simulator spreads each bin's
        flux over these pixels and the extractor integrates over the same
        ones, which makes the simulate -> extract round trip flux-conserving
        for any binning.
        """
        centres = self.wavelength_to_pixel(self.wavelength_grid())
        half = 0.5 * self.bin_width_nm() * self.pix_per_nm
        lo, hi = centres - half, centres + half
        edges_lo = np.arange(self.axis_length) - 0.5
        overlap = (np.minimum(hi[:, None], edges_lo[None, :] + 1.0)
                   - np.maximum(lo[:, None], edges_lo[None, :]))
        return np.clip(overlap, 0.0, None)
