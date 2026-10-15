"""multi_wvl -- per-wavelength photonic-lantern propagation, split into
params / bookkeeping / construct / pipeline / glue / run submodules."""
from __future__ import annotations

from .params import (
    build_lantern_geometry,
    sellmeier_n_silica,
    scale_params_to_wavelength,
)
from .bookkeeping import (
    probe_endpoint_neffs,
    infer_lantern_mode_bookkeeping,
    diagnose_mode_bookkeeping,
)
from .construct import build_and_characterize_lantern_at_wavelength
from .pipeline import WavelengthEngine, MultiWavelengthPropagationPipeline
from .glue import spectra_for_field, build_spectral_config_from_wavelength_grid
from .run import example_multiwavelength_propagation

__all__ = [
    "build_lantern_geometry", "sellmeier_n_silica", "scale_params_to_wavelength",
    "probe_endpoint_neffs", "infer_lantern_mode_bookkeeping", "diagnose_mode_bookkeeping",
    "build_and_characterize_lantern_at_wavelength",
    "WavelengthEngine", "MultiWavelengthPropagationPipeline",
    "spectra_for_field", "build_spectral_config_from_wavelength_grid",
    "example_multiwavelength_propagation",
]
