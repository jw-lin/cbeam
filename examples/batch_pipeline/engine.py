"""Waveguide setup, incident-field generation, modal projection and the batch engine.

Pure processing -- this module imports no matplotlib.
"""
from __future__ import annotations
import numpy as np
import scipy.fft as sp_fft
from scipy.spatial import Delaunay

from cbeam.waveguide import PhotonicLantern
from cbeam.propagator import Propagator, ChainPropagator

from .ifunc import to_numpy
from .constants import (
    backend, jax, jnp, _jax_device,
    default_chunk_size, default_gen_chunk_size, DEFAULT_GRID_RESOLUTION,
    default_degenerate_groups_front,
    L1, L2,
)


def _jax_fft_batch(Ef_input_batch, pad_width):
    """JIT-compiled zero-pad + FFT for a batch of pupil fields (JAX only)."""
    E_padded = jnp.pad(
        Ef_input_batch,
        ((0, 0), (pad_width, pad_width), (pad_width, pad_width)),
        mode='constant',
        constant_values=0,
    )
    return jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.ifftshift(E_padded, axes=(1, 2)),
            axes=(1, 2),
        ),
        axes=(1, 2),
    )


if backend == 'jax':
    _apply_pupil_jax_core = jax.jit(_jax_fft_batch, static_argnames=['pad_width'])


def wavelength_tag(wavelength_nm: float) -> str:
    """Filename tag matching the convention used for the 800 nm cache
    (800.0 nm -> '0800'). Adjust if your cache naming differs, or if your
    native grid isn't on round-nm values -- in that case you likely want
    an explicit {wavelength_nm: tag} mapping instead of this formula."""
    return f"{int(round(wavelength_nm)):04d}"


def build_and_characterize_lantern(p, reuse_cache=False):
    """Set up the PhotonicLantern and return a ChainPropagator.

    Mode bookkeeping: the front segment tracks every mode (only its
    degenerate pairs are grouped -- they come from the input end, whose modes
    do not depend on the mesh). The back segment skips the mode(s) that have
    become cladding modes by the junction (effective index below the
    cladding index) and treats the rest -- the core supermodes -- as one
    degenerate group. Which mode index ends up in the cladding depends on how
    the tracker labels the modes through the taper, so it is read off the
    front segment instead of being fixed in advance: a fixed choice (mode 18,
    as in docs_source/PL19.rst) is right only for some meshes, and on others
    it zeroes a core mode instead, leaving that core dark.

    reuse_cache : if True, a previously saved characterization matching the
        (port-count, wavelength) tag is loaded instead of being recomputed.
        The tag does NOT encode taper_factor / resolutions / radii, so only
        enable this when those have not changed since the cache was written.
    """
    PL_nrings = PhotonicLantern(
        p["output_positions"], p["rcores"], p["rclad"], p["rjack"],
        p["ncores"], p["nclad"], p["njack"], p["z_ex"],
        p["taper_factor"], p["core_res"], p["clad_res"], p["jack_res"],
    )

    L1_ = p.get("L1", L1)
    L2_ = p.get("L2", L2)

    cache_prefix       = "port"
    n_output_positions = str(p["n_output_positions"])
    wavelength_nm      = p["wavelength_nm"]
    tag_base = f"{n_output_positions}{cache_prefix}_{wavelength_tag(wavelength_nm)}"

    def _characterize_or_load(prop, zi, zf, tag, seed=None):
        if reuse_cache:
            try:
                prop.load(tag)
                print(f"  loaded cached characterization '{tag}'")
                return
            except (FileNotFoundError, OSError):
                print(f"  no cache for '{tag}', characterizing ...")
        if seed is not None:
            prop.load_init_conds(seed)
        prop.characterize(zi, zf, save=True, tag=tag)

    nr = p["nrings"]
    if nr not in default_degenerate_groups_front:
        raise ValueError(
            f"no default degen_groups configured for nrings={nr}; only "
            f"nrings={sorted(default_degenerate_groups_front)} is set up. Add the "
            f"input-end degenerate pairs to default_degenerate_groups_front.")
    n_modes = p["n_output_positions"] + 1

    prop1 = Propagator(p["wl"], PL_nrings, n_modes)
    # copy so an in-place mutation by the propagator can't poison the module
    # globals for the next call
    prop1.degen_groups  = [list(g) for g in default_degenerate_groups_front[nr]]
    prop1.skipped_modes = set()
    _characterize_or_load(prop1, 0, L1_, tag_base + "_front-all")

    n_junction = np.real(np.asarray(prop1.neffs[-1]))
    cladding = {k for k in range(n_modes) if n_junction[k] < p["nclad"]}
    prop2 = Propagator(p["wl"], PL_nrings, n_modes)
    prop2.degen_groups  = [[k for k in range(n_modes) if k not in cladding]]
    prop2.skipped_modes = cladding
    _characterize_or_load(prop2, L1_, L1_ + L2_, tag_base + "_back-clad", seed=prop1)

    return ChainPropagator([prop1, prop2])


def get_waveguide_properties(propagator, mesh_z=0):
    """Extract waveguide modal properties at a given z position.

    ``propagator`` is any characterized ``Propagator`` or ``ChainPropagator``
    (for a chain, the segment that contains ``mesh_z`` is used).

    The returned mesh is the physical cross-section at ``mesh_z``: a
    propagator's own ``.mesh`` is the z=0 reference mesh, which for a taper
    differs from the real one by the local scale factor (e.g. ``taper_factor``
    at the far end of a PhotonicLantern).
    """
    from wavesolve.fe_solver import construct_B  # only used here

    p_segment  = (propagator.get_prop(mesh_z) if hasattr(propagator, "get_prop")
                  else propagator)
    mesh_obj   = p_segment.make_mesh_at_z(mesh_z)
    p_segment.wvg.assign_IOR()          # ensure wvg.IOR_dict is populated
    modes      = p_segment.vs

    if len(modes.shape) == 3:
        z_idx        = np.argmin(np.abs(p_segment.zs - mesh_z))
        active_modes = modes[z_idx]
    else:
        active_modes = modes

    n_mesh_points = mesh_obj.points.shape[0]

    if active_modes.shape[0] == n_mesh_points:
        active_modes = active_modes.T

    n_modes   = active_modes.shape[0]
    points_2d = np.ascontiguousarray(mesh_obj.points[:, :2])

    B     = construct_B(mesh_obj, sparse=True)
    areas = np.array(B.diagonal())
    return {
        'mesh':          mesh_obj,
        'mesh_areas':    areas,        # diag(B); kept for lightweight diagnostics
        'B':             B,            # full FE mass matrix, for exact modal overlaps
        'points':        points_2d,
        'modes':         active_modes,
        'n_modes':       n_modes,
        'n_mesh_points': n_mesh_points,
    }


class ModalProjector:
    """Project spatial field profiles onto the modal basis with normalisation."""

    def __init__(self, wvg_props, xp=np):
        self.xp = xp
        # Batched form of Propagator.make_mode_vector / inner_product:
        #   u0[k] = conj(v_k)^T . B . E    with the full FE mass matrix B
        # (the cbeam eigenmodes are B-orthonormal, so this is a clean
        # projection; diag(B) alone is not -- v^T diag(B) v ~ 0.63, not 1).
        modes = np.asarray(wvg_props["modes"])            # (n_modes, n_pts)
        B     = wvg_props["B"]                            # sparse, host
        pm    = B.dot(modes.conj().T).T                   # (n_modes, n_pts) dense
        self.projection_matrix = xp.asarray(pm)           # -> device iff xp is jnp

    def project_batch(self, E_batch):
        """
        Project a batch of mesh-sampled electric fields onto the mode basis.

        Returns
        -------
        u0_batch : array, shape (n_fields, n_modes)
            Unit-L2-norm modal coefficients: the overlap integrals
            ``⟨vₖ | E⟩`` divided by their own L2 norm (rows that are exactly
            zero are left as zero).  This is a normalised initial condition,
            ready to hand to ``propagate``.  The absolute amount of pupil
            power that actually coupled into the tracked modes is *not*
            carried here -- it is returned separately as
            ``coupling_efficiency``.
        coupling_efficiency : array, shape (n_fields,)
            L2 norm of each projected vector *before* normalisation, in the
            field's own units.  Divided by the incident field's total
            focal-plane amplitude (``IncidentFieldGenerator.focal_plane_norm``,
            as ``BatchPropagationPipeline`` does) it becomes the fraction of
            amplitude that coupled into the tracked modes.  Retain it if you
            need output power spectra to be comparable across wavelengths --
            the normalised ``u0_batch`` alone cannot be.
        """
        u0_batch = E_batch @ self.projection_matrix.T
        coupling_efficiency = self.xp.linalg.norm(u0_batch, axis=1)
        # Avoid division by zero for all-zero inputs (e.g. a masked-out field).
        safe_norms = coupling_efficiency[:, None]
        if self.xp is np:
            safe_norms = np.where(safe_norms == 0, 1.0, safe_norms)
        else:
            safe_norms = self.xp.where(safe_norms == 0, 1.0, safe_norms)
        return u0_batch / safe_norms, coupling_efficiency


class IncidentFieldGenerator:
    """Build batches of DM-aberrated pupil fields from influence-function
    coefficients, form their focal-plane image (zero-padded FFT) and sample it
    on the waveguide's input mesh."""

    def __init__(self, p, ifunc, xp=np, pupil_template=None):
        self.p     = p
        self.ifunc = ifunc
        self.xp    = xp

        if pupil_template is not None:
            # DM mask + influence-function matrix don't depend on
            # propagation wavelength -- reuse a copy built once elsewhere
            # instead of re-reading/re-stacking ifunc.influence_function.
            self.mask_np, self.grid_size, self.num_modes, self.ifunc_matrix = pupil_template
        else:
            # to_numpy: specula keeps these on its own device (CuPy on a GPU)
            self.mask_np   = to_numpy(ifunc.mask_inf_func) > 0
            self.grid_size = self.mask_np.shape[0]
            self.num_modes = len(ifunc.influence_function)
            self.ifunc_matrix = xp.array(to_numpy(ifunc.influence_function))

    def _pad_geometry(self):
        """(pad_width, padded_size) of the symmetric zero-padded FFT grid.

        Single source of truth for the padded-grid size: the FFT paths pad by
        ``pad_width`` on each side, so the grid the interpolation weights are
        built against must be ``grid_size + 2*pad_width`` (== grid_size *
        pad_factor exactly when grid_size*(pad_factor-1) is even, which is the
        case for the bundled 400 px / pad_factor 4 ifunc -> (600, 1600))."""
        N = self.grid_size
        pad_width = (N * self.p["pad_factor"] - N) // 2
        return pad_width, N + 2 * pad_width

    @property
    def focal_plane_norm(self):
        """Square root of the total focal-plane power of any pupil field
        (unit amplitude on every mask pixel), in the same um^2 units as the
        modal overlaps: by Parseval, sum|fft2|^2 = padded_size^2 *
        sum|pupil|^2, and each focal-plane pixel is pixel_scale_um^2."""
        _, padded_size = self._pad_geometry()
        pixel_scale = self.p.get("pixel_scale_um", 1.0)
        return padded_size * np.sqrt(np.count_nonzero(self.mask_np)) * pixel_scale

    def precompute_interpolation_weights(self, mesh_points):
        """Bilinear-interpolation weights mapping the padded FFT grid onto the
        FE mesh nodes: even/odd-aware FFT centre, mesh x -> column / mesh y ->
        row (see the axis-mapping note below), plus out-of-bounds validation."""
        _, padded_size = self._pad_geometry()

        # FFT centre index: padded_size // 2 (even) or (padded_size - 1) // 2 (odd)
        if padded_size % 2 == 0:
            center_offset = padded_size / 2.0
        else:
            center_offset = (padded_size - 1) / 2.0
        
        print(f"[Interpolation Setup]")
        print(f"  Padded grid size: {padded_size}")
        print(f"  Grid parity: {'even' if padded_size % 2 == 0 else 'odd'}")
        print(f"  Center offset: {center_offset}")

        pixel_scale = self.p.get("pixel_scale_um", 1.0)
        print(f"  Pixel scale: {pixel_scale} μm/pixel")

        # Axis mapping.  cbeam stores mesh.points[:, 0] = x, [:, 1] = y (see
        # waveguide.py).  The FFT image E_focal is indexed [row, col] in the
        # standard image convention (row = y, col = x) -- the same convention
        # used by diagnose_input_psf and interpolate_output_to_grid.  So the
        # column index (ix0) is built from physical x and the row index (iy0)
        # from physical y; anything else transposes the input relative to the
        # rest of the pipeline.
        mesh_x = mesh_points[:, 0]   # physical x -> FFT column (ix0)
        mesh_y = mesh_points[:, 1]   # physical y -> FFT row    (iy0)

        # Validate mesh range
        print(f"  Mesh X range: [{mesh_x.min():.3f}, {mesh_x.max():.3f}] μm")
        print(f"  Mesh Y range: [{mesh_y.min():.3f}, {mesh_y.max():.3f}] μm")

        # Convert physical coordinates to continuous pixel indices
        # Grid coordinate = physical_coordinate / pixel_scale + center
        mesh_x_pix = (mesh_x / pixel_scale) + center_offset
        mesh_y_pix = (mesh_y / pixel_scale) + center_offset

        # Check for out-of-bounds points
        n_out_of_bounds = np.sum((mesh_y_pix < 0) | (mesh_y_pix >= padded_size) |
                                  (mesh_x_pix < 0) | (mesh_x_pix >= padded_size))
        if n_out_of_bounds > 0:
            print(f"  ⚠️  WARNING: {n_out_of_bounds} mesh points map outside FFT grid!")
            print(f"     Check pixel_scale_um and mesh extent.")

        # Bilinear interpolation weights
        iy0 = self.xp.floor(mesh_y_pix).astype(self.xp.int32)
        iy1 = iy0 + 1
        ix0 = self.xp.floor(mesh_x_pix).astype(self.xp.int32)
        ix1 = ix0 + 1

        wy1 = mesh_y_pix - iy0
        wy0 = 1.0 - wy1
        wx1 = mesh_x_pix - ix0
        wx0 = 1.0 - wx1

        # Boundary validation
        valid = ((iy0 >= 0) & (iy1 < padded_size) &
                 (ix0 >= 0) & (ix1 < padded_size))

        # Clip indices
        iy0 = self.xp.clip(iy0, 0, padded_size - 1)
        iy1 = self.xp.clip(iy1, 0, padded_size - 1)
        ix0 = self.xp.clip(ix0, 0, padded_size - 1)
        ix1 = self.xp.clip(ix1, 0, padded_size - 1)

        # Store
        self.iy0 = self.xp.asarray(iy0)
        self.iy1 = self.xp.asarray(iy1)
        self.ix0 = self.xp.asarray(ix0)
        self.ix1 = self.xp.asarray(ix1)
        # 1-D (n_pts,) weights; they broadcast against the (n_fields, n_pts)
        # sampled-value arrays in resample_to_mesh on the trailing axis.
        self.w00 = self.xp.asarray(wy0 * wx0 * valid)
        self.w01 = self.xp.asarray(wy0 * wx1 * valid)
        self.w10 = self.xp.asarray(wy1 * wx0 * valid)
        self.w11 = self.xp.asarray(wy1 * wx1 * valid)
        
        print(f"  Valid interpolation points: {np.sum(valid)} / {len(valid)}")

    def generate_field_profiles_batch(self, coeff_batch):
        """Generate complex electric field profiles from aberration coefficients."""
        opd_flat_batch   = coeff_batch @ self.ifunc_matrix
        phase_flat_batch = opd_flat_batch * (2 * self.xp.pi / self.p["wavelength_nm"])
        E_flat_batch     = self.xp.exp(1j * phase_flat_batch)

        n_fields      = coeff_batch.shape[0]
        Ef_input_batch = self.xp.zeros(
            (n_fields, self.grid_size, self.grid_size), dtype=self.xp.complex128)
        if self.xp is np:
            Ef_input_batch[:, self.mask_np] = E_flat_batch
        else:
            Ef_input_batch = Ef_input_batch.at[:, self.mask_np].set(E_flat_batch)
        return Ef_input_batch

    def apply_ef_to_waveguide_jax(self, Ef_input_batch):
        """Focal-plane image of a batch of pupil fields: pad and FFT-transform
        using JAX (runs on the selected device)."""
        pad_width, _       = self._pad_geometry()
        Ef_input_batch_dev = jax.device_put(Ef_input_batch, _jax_device)
        return _apply_pupil_jax_core(Ef_input_batch_dev, pad_width)

    def apply_ef_to_waveguide(self, Ef_input_batch):
        """Focal-plane image of a batch of pupil fields: pad and FFT-transform
        using SciPy (CPU, multithreaded)."""
        N                     = self.grid_size
        pad_width, padded_size = self._pad_geometry()

        E_padded = np.zeros(
            (Ef_input_batch.shape[0], padded_size, padded_size),
            dtype=np.complex128)
        E_padded[:, pad_width:pad_width+N, pad_width:pad_width+N] = \
            np.asarray(Ef_input_batch)

        return sp_fft.fftshift(
            sp_fft.fft2(
                sp_fft.ifftshift(E_padded, axes=(1, 2)),
                axes=(1, 2),
                workers=-1,
            ),
            axes=(1, 2),
        )

    def resample_to_mesh(self, E_focal_batch):
        """Bilinear interpolation from the padded FFT grid onto the FE mesh."""
        val00 = E_focal_batch[:, self.iy0, self.ix0]
        val01 = E_focal_batch[:, self.iy0, self.ix1]
        val10 = E_focal_batch[:, self.iy1, self.ix0]
        val11 = E_focal_batch[:, self.iy1, self.ix1]
        return (val00 * self.w00 + val01 * self.w01 +
                val10 * self.w10 + val11 * self.w11)

    # earlier names, kept for existing callers
    apply_ef_to_lantern     = apply_ef_to_waveguide
    apply_ef_to_lantern_jax = apply_ef_to_waveguide_jax


class BatchPropagationPipeline:
    """
    Orchestrates batch propagation of modal fields through a waveguide.

    Works with any characterized ``Propagator`` or ``ChainPropagator`` (a
    photonic lantern, a fiber, a coupler, ...): the input fields are sampled on
    its mesh at the start of the characterized range, projected onto the modes
    there, propagated, and the output fields are rebuilt from the modes at the
    end of the range.  The input fields themselves are DM-aberrated pupils
    imaged onto the input face (``IncidentFieldGenerator``), so ``p`` and
    ``ifunc`` describe the telescope side.

    Parameters
    ----------
    propagator :           characterized Propagator or ChainPropagator
    p :                    simulation parameter dict (``wavelength_nm``,
                           ``pad_factor``, ``pixel_scale_um`` for the pupil
                           and its focal-plane image)
    ifunc :                influence-function object
    field_gen_on_gpu :     bool, optional (default False)
        When True *and* the JAX backend is active, field generation runs on
        the GPU.  When False (default), field generation always uses
        NumPy/SciPy on the CPU, which avoids GPU memory pressure during the
        OPD → pupil → FFT → projection phase.
        Can be overridden per call via the ``use_gpu`` keyword of
        ``generate_batch_modal_coefficients()``.
    field_gen_chunk_size : int or None, optional (default 64)
        Number of fields processed in each chunk of
        ``generate_batch_modal_coefficients()``.  The large intermediate
        arrays (OPD batch, padded FFT grid, mesh-sampled fields) are
        allocated, used, and released chunk by chunk, capping peak memory
        to O(chunk_size) independently of the total batch size and
        independently of which backend is used.  Set to None to disable
        chunking (process the whole batch at once).
        Can be overridden per call via the ``chunk_size`` keyword.
    """

    def __init__(self, propagator, p, ifunc,
                 field_gen_on_gpu=False,
                 field_gen_chunk_size=default_gen_chunk_size,
                 field_gen_pupil_template=None):
        
        self.propagator = propagator
        self.p          = p

        # The *propagation* xp follows the global backend.
        self.xp = jnp if backend == 'jax' else np

        # Default chunk size for generate_batch_modal_coefficients().
        # None means "no chunking" (process the whole batch at once).
        self.field_gen_chunk_size = field_gen_chunk_size

        # -----------------------------------------------------------------
        # Decide the default array module for field generation.
        # -----------------------------------------------------------------
        self._field_gen_on_gpu_default = (backend == 'jax') and field_gen_on_gpu

        # Always build the numpy generators (cheap, host memory only).
        self._field_gen_pupil_template = field_gen_pupil_template
        self._field_gen_np   = IncidentFieldGenerator(
            p, ifunc, xp=np, pupil_template=field_gen_pupil_template)

        
        self._modal_proj_np  = None   # built after weight precomputation below

        # Build JAX generators only if explicitly requested.
        self._field_gen_jax  = None
        self._modal_proj_jax = None
        self._ifunc_ref      = ifunc  # kept for lazy JAX init

        # input / output planes: the ends of the characterized range
        zs = propagator.zs
        self.z_in, self.z_out = float(zs[0]), float(zs[-1])
        wvg_props_input = get_waveguide_properties(propagator, mesh_z=self.z_in)
        self.wvg_props_input  = wvg_props_input
        self.wvg_props_output = get_waveguide_properties(propagator, mesh_z=self.z_out)

        self._field_gen_np.precompute_interpolation_weights(
            wvg_props_input['points'])
        self._modal_proj_np = ModalProjector(wvg_props_input, xp=np)

        if backend == 'jax' and field_gen_on_gpu:
            self._build_jax_field_gen()

        # Delaunay cache for output interpolation; filled lazily by
        # interpolate_output_to_grid() the first time each resolution is used.
        self._delaunay_cache = {}

    @property
    def prop12(self):
        """Earlier name of ``propagator``, kept for existing callers."""
        return self.propagator

    # ------------------------------------------------------------------
    def _build_jax_field_gen(self):
        """Lazily initialise the JAX field-generation objects."""
        if self._field_gen_jax is not None:
            return
        self._field_gen_jax = IncidentFieldGenerator(
            self.p, self._ifunc_ref, xp=jnp)
        self._field_gen_jax.precompute_interpolation_weights(
            jnp.asarray(self.wvg_props_input['points'],
                        dtype=jnp.float64, device=_jax_device))
        self._modal_proj_jax = ModalProjector(self.wvg_props_input, xp=jnp)

    # ------------------------------------------------------------------
    def enable_gpu_field_gen(self):
        """
        Lazily initialise the JAX field-generation objects after construction.

        Call this if the pipeline was built with ``field_gen_on_gpu=False``
        but you later want to use ``use_gpu=True`` in some calls.
        """
        if backend != 'jax':
            raise RuntimeError(
                "enable_gpu_field_gen() requires the JAX backend.")
        self._build_jax_field_gen()
        print("JAX field-generation objects initialised on device:", _jax_device)

    # ------------------------------------------------------------------
    def _precompute_delaunay_grid(self, grid_resolution):
        """Precompute Delaunay triangulation and barycentric weights."""
        mesh_pts = self.wvg_props_output['mesh'].points
        x_min, x_max = mesh_pts[:, 0].min(), mesh_pts[:, 0].max()
        y_min, y_max = mesh_pts[:, 1].min(), mesh_pts[:, 1].max()

        plot_x = np.linspace(x_min, x_max, grid_resolution)
        plot_y = np.linspace(y_min, y_max, grid_resolution)
        X_plot, Y_plot = np.meshgrid(plot_x, plot_y)
        grid_pts = np.column_stack((X_plot.ravel(), Y_plot.ravel()))

        tri        = Delaunay(mesh_pts[:, :2])
        v_simplex  = tri.find_simplex(grid_pts)
        valid_grid = v_simplex >= 0

        v0 = np.zeros(grid_pts.shape[0], dtype=np.int32)
        v1 = np.zeros(grid_pts.shape[0], dtype=np.int32)
        v2 = np.zeros(grid_pts.shape[0], dtype=np.int32)
        w0 = np.zeros(grid_pts.shape[0], dtype=np.float64)
        w1 = np.zeros(grid_pts.shape[0], dtype=np.float64)
        w2 = np.zeros(grid_pts.shape[0], dtype=np.float64)

        if np.any(valid_grid):
            valid_simplices = v_simplex[valid_grid]
            v0[valid_grid]  = tri.simplices[valid_simplices, 0]
            v1[valid_grid]  = tri.simplices[valid_simplices, 1]
            v2[valid_grid]  = tri.simplices[valid_simplices, 2]

            transform = tri.transform[valid_simplices]
            r3        = tri.simplices[valid_simplices, 2].astype(np.intp)
            delta     = grid_pts[valid_grid] - tri.points[r3]
            bary      = np.einsum('nij,nj->ni', transform[:, :2], delta)
            w0[valid_grid] = bary[:, 0]
            w1[valid_grid] = bary[:, 1]
            w2[valid_grid] = 1.0 - bary[:, 0] - bary[:, 1]

        return X_plot, Y_plot, v0, v1, v2, w0, w1, w2, valid_grid

    # ------------------------------------------------------------------
    def interpolate_output_to_grid(self, E_output_batch,
                                   grid_resolution=DEFAULT_GRID_RESOLUTION):
        """Interpolate the output *intensity* |E|**2 onto a regular grid (CPU).

        Takes complex output fields on the mesh, returns
        ``(intensity_2d_batch, X_plot, Y_plot)`` where intensity_2d_batch has
        shape (n_fields, grid_resolution, grid_resolution) and is real |E|**2
        (phase is not carried through)."""
        if grid_resolution not in self._delaunay_cache:
            self._delaunay_cache[grid_resolution] = \
                self._precompute_delaunay_grid(grid_resolution)
        X_plot, Y_plot, v0, v1, v2, w0, w1, w2, valid_grid = \
            self._delaunay_cache[grid_resolution]

        E_np            = np.asarray(E_output_batch)
        intensity_batch = np.abs(E_np) ** 2

        flat_interpolated = (
            intensity_batch[:, v0] * w0 +
            intensity_batch[:, v1] * w1 +
            intensity_batch[:, v2] * w2
        )
        flat_interpolated[:, ~valid_grid] = 0.0

        n_fields          = E_np.shape[0]
        intensity_2d_batch = flat_interpolated.reshape(
            n_fields, grid_resolution, grid_resolution)
        return intensity_2d_batch, X_plot, Y_plot

    # ------------------------------------------------------------------
    def _process_chunk(self, coeff_chunk, fg, proj, use_gpu):
        """Run the field-generation pipeline on a single coefficient chunk.

        Returns ``(u0_chunk, eff_chunk)`` as plain numpy arrays of shape
        ``(chunk_size, n_modes)`` and ``(chunk_size,)`` respectively.
        ``eff_chunk`` is the input coupling efficiency (amplitude, not power)
        for each field: the L2 norm of its projected vector *before*
        normalisation, divided by the field's total focal-plane amplitude;
        ``u0_chunk`` itself is unit-norm.
        All intermediate arrays are local to this call and are released
        when it returns.
        """
        Ef_input  = fg.generate_field_profiles_batch(coeff_chunk)
        if use_gpu:
            E_focal = fg.apply_ef_to_waveguide_jax(Ef_input)
        else:
            E_focal = fg.apply_ef_to_waveguide(Ef_input)
        Ef_focal_mesh   = fg.resample_to_mesh(E_focal)
        u0_chunk, eff = proj.project_batch(Ef_focal_mesh)
        return np.asarray(u0_chunk), np.asarray(eff) / fg.focal_plane_norm

    # ------------------------------------------------------------------
    def generate_batch_modal_coefficients(self, aberration_coeff_batch,
                                          use_gpu=None,
                                          chunk_size=None,
                                          return_coupling_efficiency=False):
        """
        Generate batch modal coefficients from aberration configurations.

        The input batch is split into chunks of *chunk_size* fields.  For
        each chunk the full pipeline (OPD → pupil → FFT → mesh resample →
        modal projection) is executed and its result accumulated in a plain
        numpy array on the host before the next chunk is started.  This
        limits peak memory to O(chunk_size) **regardless of the total batch
        size and regardless of which backend is used**.

        Parameters
        ----------
        aberration_coeff_batch : array-like, shape (n_configs, n_active_modes)
            Each row is one aberration configuration.  Zeros are appended
            automatically if n_active_modes < n_modes_total.
        use_gpu : bool or None, optional
            Override the instance-level ``field_gen_on_gpu`` setting.

            - ``None``  (default) — use the instance default.
            - ``False`` — force CPU/NumPy for this call.
            - ``True``  — force GPU/JAX for this call (requires JAX backend
              and initialised JAX generators).
        chunk_size : int or None, optional
            Number of fields per processing chunk.  Overrides the instance-
            level ``field_gen_chunk_size`` for this call only.

            - ``None``  — use the instance default (``field_gen_chunk_size``).
            - Positive int — use that chunk size.
            - ``0`` or negative — disable chunking (process whole batch).
        return_coupling_efficiency : bool, optional (default False)
            When True, also return the per-field input coupling efficiency
            (see below).  Kept opt-in so existing callers that unpack a
            single return value keep working.

        Returns
        -------
        u0_batch : np.ndarray, shape (n_configs, n_modes), complex128
            Normalised (unit-L2-norm) modal coefficients, always a plain
            numpy array.
        eff_batch : np.ndarray, shape (n_configs,), float64
            Only returned when ``return_coupling_efficiency=True``.  The
            fraction of focal-plane field amplitude that coupled into the
            tracked input modes (0..1; ``eff_batch**2`` is the power fraction).  ``u0_batch`` alone discards this, so any power
            spectrum that must stay comparable across fields or wavelengths
            has to be scaled by ``eff_batch**2`` (power) / ``eff_batch``
            (amplitude).
        """
        # ------------------------------------------------------------------
        # Resolve backend choice.
        # ------------------------------------------------------------------
        if use_gpu is None:
            _use_gpu = self._field_gen_on_gpu_default
        else:
            _use_gpu = bool(use_gpu)

        if _use_gpu and backend != 'jax':
            raise RuntimeError(
                "use_gpu=True requires the JAX backend "
                "(set CBEAM_BACKEND=jax before importing cbeam).")
        if _use_gpu and self._field_gen_jax is None:
            raise RuntimeError(
                "JAX field generators have not been initialised.  "
                "Either construct BatchPropagationPipeline with "
                "field_gen_on_gpu=True, or call "
                "pipeline.enable_gpu_field_gen() first.")

        fg   = self._field_gen_jax  if _use_gpu else self._field_gen_np
        proj = self._modal_proj_jax if _use_gpu else self._modal_proj_np

        # ------------------------------------------------------------------
        # Resolve chunk size.
        # ------------------------------------------------------------------
        if chunk_size is None:
            _chunk_size = self.field_gen_chunk_size
        elif chunk_size <= 0:
            _chunk_size = None   # explicit disable
        else:
            _chunk_size = int(chunk_size)

        # ------------------------------------------------------------------
        # Prepare the full coefficient array (host numpy first, then
        # optionally moved to device inside _process_chunk).
        # ------------------------------------------------------------------
        coeff_np = np.asarray(aberration_coeff_batch, dtype=np.float64)
        if coeff_np.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {coeff_np.shape}")

        n_configs, n_active_modes = coeff_np.shape
        n_modes_total = fg.num_modes

        if n_active_modes < n_modes_total:
            padded = np.zeros((n_configs, n_modes_total), dtype=np.float64)
            padded[:, :n_active_modes] = coeff_np
            coeff_np = padded

        # ------------------------------------------------------------------
        # Process in chunks.
        # ------------------------------------------------------------------
        if _chunk_size is None or _chunk_size >= n_configs:
            # Single pass — no chunking overhead.
            coeff_chunk = self._to_device(coeff_np, _use_gpu)
            u0_batch, eff_batch = self._process_chunk(
                coeff_chunk, fg, proj, _use_gpu)
            if return_coupling_efficiency:
                return u0_batch, eff_batch
            return u0_batch

        print(f"  Generating modal coefficients for {n_configs} fields "
              f"(chunk_size={_chunk_size}, "
              f"device={'GPU' if _use_gpu else 'CPU'}) ...", flush=True)

        u0_chunks  = []
        eff_chunks = []
        for start in range(0, n_configs, _chunk_size):
            end         = min(start + _chunk_size, n_configs)
            coeff_chunk = self._to_device(coeff_np[start:end], _use_gpu)
            u0_chunk, eff_chunk = self._process_chunk(
                coeff_chunk, fg, proj, _use_gpu)
            u0_chunks.append(u0_chunk)
            eff_chunks.append(eff_chunk)
            print(f"    field-gen chunk {start}–{end-1} done.", flush=True)

        print("  Field generation complete.")
        u0_batch = np.concatenate(u0_chunks, axis=0)
        if return_coupling_efficiency:
            return u0_batch, np.concatenate(eff_chunks, axis=0)
        return u0_batch

    # ------------------------------------------------------------------
    @staticmethod
    def _to_device(coeff_np, use_gpu):
        """Move a numpy coefficient array to the appropriate device."""
        if use_gpu:
            return jnp.asarray(coeff_np, dtype=jnp.float64, device=_jax_device)
        return coeff_np   # already numpy, no copy needed

    # ------------------------------------------------------------------
    def propagate_batch(self, u0_batch, chunk_size=None):
        """
        Propagate a batch of modal coefficient vectors through the waveguide.

        For the JAX backend the batch is split into chunks of *chunk_size*
        fields.  Each chunk is propagated, its output is immediately moved
        to host memory, and only then is the next chunk started, keeping
        peak GPU memory O(chunk_size) rather than O(n_batch).

        Parameters
        ----------
        u0_batch   : np.ndarray, shape (n_fields, n_modes)
        chunk_size : int or None
            If None a default is chosen:
              - JAX backend:   ``constants.default_chunk_size`` (10000;
                lower it if GPU memory runs out)
              - numpy backend: full batch (no chunking needed)

        Returns
        -------
        uf_batch  : np.ndarray (n_fields, n_modes)
        zs        : array of z values of the last chunk's trajectory (on
                    numpy the adaptive steps differ from chunk to chunk; on
                    jax only the segment endpoints are saved)
        us_batch  : np.ndarray (n_fields, n_z, n_modes) or None
                    Full trajectory — only populated for single-chunk runs
                    to avoid excessive memory use.
        """
        n_fields = u0_batch.shape[0]

        if chunk_size is None:
            chunk_size = default_chunk_size if backend == 'jax' else n_fields

        print(f"  Propagating {n_fields} fields "
              f"(chunk_size={chunk_size}) ...", flush=True)

        uf_chunks     = []
        zs_last       = None
        us_grid_single = None   # full trajectory, kept only for single-chunk runs
        multi_chunk   = (n_fields > chunk_size)

        for start in range(0, n_fields, chunk_size):
            end   = min(start + chunk_size, n_fields)
            chunk = u0_batch[start:end]

            if backend == 'jax':
                chunk = jnp.asarray(chunk, dtype=jnp.complex128,
                                    device=_jax_device)

            zs, us_grid, uf = self.propagator.propagate(chunk)

            uf_chunks.append(np.asarray(uf))
            zs_last = np.asarray(zs)

            if not multi_chunk:
                # single-chunk path: this loop body runs exactly once
                us_grid_single = np.asarray(us_grid)

            print(f"    propagation chunk {start}–{end-1} done.", flush=True)

        uf_batch = np.concatenate(uf_chunks, axis=0)

        if multi_chunk:
            us_batch = None
            print("  (us_batch not returned for multi-chunk runs to save memory)")
        else:
            # propagator.propagate returns the trajectory as (n_z, n_fields, n_modes);
            # transpose to the (n_fields, n_z, n_modes) layout this method promises.
            us_batch = (np.transpose(us_grid_single, (1, 0, 2))
                        if us_grid_single.ndim == 3 else us_grid_single)

        print("  Batch propagation complete.")
        return uf_batch, zs_last, us_batch

    # ------------------------------------------------------------------
    def reconstruct_batch_output_fields(self, uf_batch):
        """Reconstruct spatial output fields from modal coefficients (CPU)."""
        modes_out = self.wvg_props_output['modes']
        return np.asarray(uf_batch) @ modes_out

