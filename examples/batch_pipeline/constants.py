"""Backend selection, module constants and the tuned mode-bookkeeping tables."""
from __future__ import annotations
import os
from cbeam.backend import get_backend, get_jax_device
from cbeam.waveguide import hex_ring_positions

backend = get_backend()
default_chunk_size = 10000
default_gen_chunk_size = 64

# -- optional JAX imports ------------------------------------------------
if backend == 'jax':
    import jax
    import jax.numpy as jnp
    _jax_device = get_jax_device()
else:
    jax = None
    jnp = None
    _jax_device = None

L1 = 50000
L2 = 50000

DEFAULT_WAVELENGTH_UM   = 0.8
DEFAULT_GRID_RESOLUTION = 400
CALIB_GRID_RESOLUTION   = 400    # grid the mode profile is resampled onto for core-centre calibration
DEFAULT_SUBPIXEL_N      = 5      # side of the NxN sub-pixel sampling window
N_RINGS   = 3
N_SIGNALS = len(hex_ring_positions(N_RINGS, 1.0))   # lantern output cores; 19 for N_RINGS=3

# degenerate mode pairs at the lantern input (z=0), per number of rings: the
# modes there do not depend on the mesh, so these pairs are fixed. The skipped
# (cladding) modes of the back segment are found by build_and_characterize_lantern.
default_degenerate_groups_front = {
    3: [[1,2],[3,4],[6,7],[8,9],[10,11],[12,13],[15,16]],
}
