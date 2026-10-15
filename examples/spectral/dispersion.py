# Split out of the former single-file spectral_extraction_module.py.
"""DispersionModel: the one wavelength <-> detector-pixel mapping.

Both halves of the pipeline need this relation and they must agree exactly:
``SpectralImageSimulator`` uses ``wavelength_to_pixel`` to lay each fiber
trace down on the detector, and ``SpectralExtractor`` must invert *that same*
relation when it resamples the along-trace cut back onto the wavelength grid.

Before this was factored out, the simulator applied
``(w - lambda_min) * pix_per_nm + centre_offset`` while the extractor used an
unrelated ``(w - lambda_min) * detector_width / (lambda_max - lambda_min)`` --
so the extractor sampled the wrong detector columns and the recovered spectra
were mostly empty.  Keeping both directions here stops them drifting again.
"""
from __future__ import annotations
import numpy as np

from .config import SpectralConfig


class DispersionModel:
    """Linear dispersion: wavelength maps to a pixel coordinate along the
    dispersion axis, with the band centre pinned to the detector centre."""

    def __init__(self, config: SpectralConfig):
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
