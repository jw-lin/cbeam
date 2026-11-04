"""batch_pipeline -- photonic-lantern batch propagation, split into
constants / config / ifunc / engine / analysis / plots / benchmark / run
submodules.

specula is optional: everywhere the pipeline needs DM influence-function data
(field generation in ``engine.IncidentFieldGenerator`` /
``engine.BatchPropagationPipeline``, and the demo entry point in
``run.main_batch_propagation``), it takes an already-loaded ``ifunc`` object
as a plain argument. ``ifunc.load_ifunc`` loads one as specula's ``IFunc``
when specula is installed and as the plain-numpy ``ifunc.SimpleIFunc``
otherwise, with bit-identical results -- see examples/BatchedPiston.ipynb.
"""
from __future__ import annotations

from .constants import (
    backend,
    default_chunk_size,
    default_gen_chunk_size,
    L1,
    L2,
    DEFAULT_WAVELENGTH_UM,
    DEFAULT_GRID_RESOLUTION,
    CALIB_GRID_RESOLUTION,
    DEFAULT_SUBPIXEL_N,
    N_RINGS,
    N_SIGNALS,
)
from .config import (
    get_simulation_parameters,
    create_sparse_aberration_configs_mono,
    create_random_aberration_configs,
    create_ramp_aberration_configs,
)
from .ifunc import (
    SimpleIFunc,
    load_ifunc,
    center_ifunc_mask,
)
from .engine import (
    wavelength_tag,
    build_and_characterize_lantern,
    get_waveguide_properties,
    ModalProjector,
    IncidentFieldGenerator,
    BatchPropagationPipeline,
)
from .analysis import (
    calibrate_subpixel_centers,
    map_evaluated_to_ideal_geometry,
    collect_subpixel_signals,
    detect_centers_from_grid,
    batch_collect_subpixel_signals,
)
from .plots import (
    diagnose_input_psf,
    visualize_batch_output,
    display_hex_grid_plots,
    visualize_batch_hex_grid_signals,
    batch_statistics,
)
from .benchmark import (
    benchmark_propagation,
    make_random_modal_batch,
    time_one_at_a_time,
    time_batched,
)
from .run import main_batch_propagation

__all__ = [
    "backend", "default_chunk_size", "default_gen_chunk_size", "L1", "L2",
    "DEFAULT_WAVELENGTH_UM", "DEFAULT_GRID_RESOLUTION", "CALIB_GRID_RESOLUTION",
    "DEFAULT_SUBPIXEL_N", "N_RINGS", "N_SIGNALS",
    "get_simulation_parameters", "create_sparse_aberration_configs_mono",
    "create_random_aberration_configs", "create_ramp_aberration_configs",
    "SimpleIFunc", "load_ifunc", "center_ifunc_mask",
    "wavelength_tag", "build_and_characterize_lantern", "get_waveguide_properties",
    "ModalProjector", "IncidentFieldGenerator", "BatchPropagationPipeline",
    "calibrate_subpixel_centers", "map_evaluated_to_ideal_geometry",
    "collect_subpixel_signals", "detect_centers_from_grid",
    "batch_collect_subpixel_signals",
    "diagnose_input_psf", "visualize_batch_output", "display_hex_grid_plots",
    "visualize_batch_hex_grid_signals", "batch_statistics",
    "benchmark_propagation", "make_random_modal_batch",
    "time_one_at_a_time", "time_batched",
    "main_batch_propagation",
]
