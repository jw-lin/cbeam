# Auto-split from the former single-file multi_wvl_pipeline.py.
"""MultiWavelengthPropagationPipeline: one wavelength engine built/released at a time."""
from __future__ import annotations
import gc
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from batch_pipeline.constants import default_gen_chunk_size, N_SIGNALS
from batch_pipeline.engine import BatchPropagationPipeline
from cbeam.propagator import ChainPropagator

from .params import build_lantern_geometry
from .construct import build_and_characterize_lantern_at_wavelength


def _force_release_memory() -> None:
    """
    Best-effort release of both Python- and Julia-side memory between
    wavelengths. gc.collect() handles Python objects (breaking reference
    cycles CPython's refcounting alone won't catch). cbeam's Julia
    backend (accessed via PythonCall/juliacall -- see the
    FEval.transverse_gradient calls in compute_cmats) has its own
    garbage collector that isn't triggered just because Python drops its
    references to the wrapping objects, so nudge it too. Safe no-op if
    juliacall isn't importable or exposes a different entry point than
    assumed here.
    """
    gc.collect()

    # JAX: drop compiled-executable / tracing caches and let the freed device
    # buffers go back to the driver (works together with
    # XLA_PYTHON_CLIENT_PREALLOCATE=false set in cbeam.backend).  No-op on the
    # numpy backend.
    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass

    gc.collect()

    try:
        from juliacall import Main as _jl
        _jl.GC.gc()
    except Exception:
        pass



@dataclass
class WavelengthEngine:
    """One fully-built single-wavelength propagation pipeline."""
    wavelength_nm: float
    p_lambda: dict
    prop12: ChainPropagator
    pipeline: BatchPropagationPipeline



class MultiWavelengthPropagationPipeline:
    """
    Manages per-wavelength propagation through the lantern across a set
    of native wavelengths, WITHOUT holding every wavelength's heavy state
    (FEM mesh, mode-field trajectory, coupling-matrix trajectory) resident
    in memory at once.

    Usage
    -----
        mwp = MultiWavelengthPropagationPipeline(
            base_params=get_simulation_parameters(),
            ifunc=ifunc,
            native_wavelengths_nm=np.linspace(700, 900, 21),  # coarse solve grid
        )
        power, wl_out = mwp.get_power_spectra(
            coeff_matrix, output_wavelengths_nm=np.linspace(700, 900, 200)
        )  # power: (200, n_fields, n_fibers)

    Memory model
    ------------
    Each wavelength's ChainPropagator carries a mode-field trajectory
    (`vs`, shape roughly n_z_steps x n_modes x n_mesh_points, complex128)
    and a coupling-matrix trajectory (`cmats`, n_z_steps x n_modes x
    n_modes, complex128) for both the front and back segments. Depending
    on mesh size and adaptive step count these can individually reach the
    GB range. An earlier version of this class built and kept a
    BatchPropagationPipeline for every native wavelength in self.engines
    simultaneously -- fine for a handful of wavelengths, but it scales
    memory linearly with n_wavelengths and can OOM-kill the process
    (observed as a silent kernel crash, no Python traceback, after "a
    few" wavelengths loaded) for larger native grids.

    This version builds one wavelength's engine at a time inside
    propagate_batch(), extracts only the small result actually needed
    (uf_batch sliced to n_fibers complex numbers per field -- not the
    underlying GB-scale trajectory arrays), and explicitly drops the
    engine + forces garbage collection (Python and, best-effort, Julia --
    see _force_release_memory()) before moving to the next wavelength.
    Peak memory is therefore roughly O(1 wavelength) rather than
    O(n_wavelengths).

    Set cache_engines=True to opt back into keeping every engine resident
    (e.g. if you're propagating many different aberration batches against
    the same small native grid and want to avoid rebuilding each time) --
    only do this if you've confirmed your native grid is small enough,
    and your machine has enough RAM, to hold all of them at once.

    Notes on cost
    -------------
    Each native wavelength needs its own local-mode-basis characterization
    (see build_and_characterize_lantern_at_wavelength -- solved once and
    cached to disk; a fresh characterize() can take ~800 s (~13 min) on
    comparable z-extents, though loading a cached one is normally fast).
    Keep native_wavelengths_nm coarse (tens of points, denser near known
    mode-degeneracy/avoided-crossing regions -- see cbeam's compute_neffs()
    sanity check) and use output_wavelengths_nm for the finer grid you
    actually want in the dispersed spectra -- see
    interpolate_complex_spectra().
    """

    def __init__(
        self,
        base_params: dict,
        ifunc,
        native_wavelengths_nm: Sequence[float],
        n_fibers: int = N_SIGNALS,
        field_gen_on_gpu: bool = False,
        field_gen_chunk_size: Optional[int] = default_gen_chunk_size,
        cache_engines: bool = False,
        auto_mode_bookkeeping: bool = False,
        verbose: bool = True,
    ):
        self.base_params = base_params
        self.ifunc = ifunc
        self.native_wavelengths_nm = np.asarray(
            sorted(native_wavelengths_nm), dtype=np.float64)
        self.n_fibers = n_fibers
        self.field_gen_on_gpu = field_gen_on_gpu
        self.field_gen_chunk_size = field_gen_chunk_size
        self.cache_engines = cache_engines
        # Passed to build_and_characterize_lantern_at_wavelength: when a
        # wavelength has to be characterised from scratch, derive its
        # degen_groups / skipped_modes from an n_eff probe instead of
        # reusing the 800 nm hard-coded config. No effect on cached
        # wavelengths.
        self.auto_mode_bookkeeping = auto_mode_bookkeeping
        self.verbose = verbose

        # DM mask / influence-function matrix -- wavelength independent,
        # so num_modes is available immediately without building any
        # engine (contrast with the earlier version, where you had to
        # reach into mwp.engines[wl].pipeline._field_gen_np.num_modes).
        self.num_modes = len(ifunc.influence_function)

        # Wavelength-independent lantern geometry, built once and shared
        # by every per-wavelength Propagator.
        self._shared_PL_N = build_lantern_geometry(base_params)

        # Wavelength-independent pupil data (DM mask + ifunc_matrix),
        # captured from whichever engine is built first and reused for
        # every subsequent one. See IncidentFieldGenerator.pupil_template.
        self._pupil_template = None

        # Only populated per-wavelength if cache_engines=True. Empty by
        # default -- engines are built, used, and discarded inside
        # propagate_batch() instead of being pre-built here.
        self.engines: Dict[float, WavelengthEngine] = {}

        if verbose:
            print(f"[MultiWavelength] Configured for {len(self.native_wavelengths_nm)} "
                  f"native wavelengths "
                  f"({self.native_wavelengths_nm.min():.1f}-"
                  f"{self.native_wavelengths_nm.max():.1f} nm). Engines are built "
                  f"and released one wavelength at a time during propagate_batch() "
                  f"(cache_engines={cache_engines}).")

    # ------------------------------------------------------------------
    def _build_engine(self, wl: float) -> WavelengthEngine:
        """Build a single wavelength's ChainPropagator + BatchPropagationPipeline."""
        p_lambda, prop12 = build_and_characterize_lantern_at_wavelength(
            self.base_params, wl, PL_N=self._shared_PL_N,
            auto_mode_bookkeeping=self.auto_mode_bookkeeping, verbose=self.verbose)

        try:
            # field_gen_pupil_template reuses the wavelength-independent
            # DM mask / influence-function matrix built for the first
            # wavelength instead of rebuilding it for every one --
            # requires the field_gen_pupil_template patch to
            # IncidentFieldGenerator / BatchPropagationPipeline in
            # batch_propagation_pipeline.py.
            pipeline = BatchPropagationPipeline(
                prop12, p_lambda, self.ifunc,
                field_gen_on_gpu=self.field_gen_on_gpu,
                field_gen_chunk_size=self.field_gen_chunk_size,
                field_gen_pupil_template=self._pupil_template,
            )
        except TypeError:
            # Unpatched batch_propagation_pipeline.py -- fall back to the
            # original constructor (still correct, just redoes the
            # pupil-data load/stack for every wavelength).
            pipeline = BatchPropagationPipeline(
                prop12, p_lambda, self.ifunc,
                field_gen_on_gpu=self.field_gen_on_gpu,
                field_gen_chunk_size=self.field_gen_chunk_size,
            )

        if self._pupil_template is None:
            fg = pipeline._field_gen_np
            self._pupil_template = (fg.mask_np, fg.grid_size, fg.num_modes, fg.ifunc_matrix)

        return WavelengthEngine(float(wl), p_lambda, prop12, pipeline)

    # ------------------------------------------------------------------
    def propagate_batch(
        self,
        aberration_coeff_batch: np.ndarray,
        use_gpu: Optional[bool] = None,
        gen_chunk_size: Optional[int] = None,
        prop_chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Propagate one batch of aberration configurations through the
        lantern at every native wavelength, building and (unless
        cache_engines=True) discarding each wavelength's engine in turn.

        Parameters
        ----------
        aberration_coeff_batch : ndarray, shape (n_fields, n_active_modes)
            Same convention as BatchPropagationPipeline.generate_batch_modal_coefficients.
        use_gpu, gen_chunk_size, prop_chunk_size : passed straight through
            to each per-wavelength pipeline's generate_batch_modal_coefficients
            / propagate_batch calls.

        Returns
        -------
        spectra_complex : ndarray, shape (n_wavelengths_native, n_fields, n_fibers), complex128
            Complex per-fiber modal coefficient at each native wavelength,
            *scaled by the per-field input coupling efficiency* so that
            np.abs(...)**2 is a throughput-bearing flux that stays
            comparable across fields and across wavelengths.

            generate_batch_modal_coefficients() renormalises every input
            field to unit L2 norm, which by itself discards how much of the
            focal-plane PSF actually coupled into the lantern (this varies
            strongly as an aberration walks the PSF off the multimode
            face, and with wavelength as the PSF breathes relative to
            rclad). We recover that here by multiplying each field's
            channel amplitudes by its coupling efficiency `eff` (the L2
            norm of the projected field *before* that renormalisation).
            `eff` is a consistent relative measure, not an absolute
            "fraction of incident power" -- use it for comparisons, not as
            an absolute transmission.

            Power is NOT taken here -- phase is preserved in case you want
            it (e.g. to sanity-check mode purity); take np.abs(...)**2 for
            detector-plane flux.
        """
        n_wl = len(self.native_wavelengths_nm)
        n_fields = np.asarray(aberration_coeff_batch).shape[0]

        spectra_complex = np.zeros((n_wl, n_fields, self.n_fibers), dtype=np.complex128)

        for i, wl in enumerate(self.native_wavelengths_nm):
            if self.verbose:
                print(f"[MultiWavelength] Propagating at {wl:.2f} nm ({i + 1}/{n_wl}) ...")

            engine = self.engines.get(float(wl))
            if engine is None:
                engine = self._build_engine(wl)
                if self.cache_engines:
                    self.engines[float(wl)] = engine

            u0_batch, eff_batch = engine.pipeline.generate_batch_modal_coefficients(
                aberration_coeff_batch, use_gpu=use_gpu, chunk_size=gen_chunk_size,
                return_coupling_efficiency=True,
            )

            uf_batch, _, _ = engine.pipeline.propagate_batch(u0_batch, chunk_size=prop_chunk_size)
            # Convert from FEM-eigenmode basis to per-core channel basis

            # shape (n_fields, n_modes) → (n_fields, n_fibers) in per-core order
            uf_channel = np.stack([
                engine.prop12.to_channel_basis(uf_batch[k])
                for k in range(uf_batch.shape[0])
            ])

            # Fold the discarded input coupling efficiency back in: u0_batch
            # was renormalised to unit norm, propagation is (near-)unitary,
            # so the physical channel amplitude scales linearly with the
            # amount of pupil field that actually coupled in -- eff_batch.
            eff_batch = np.asarray(eff_batch, dtype=np.float64).reshape(-1)
            uf_channel = uf_channel * eff_batch[:, None]

            spectra_complex[i, :, :] = uf_channel[:, :self.n_fibers]

            if not self.cache_engines:
                # Drop the heavy per-wavelength state (mesh, mode-field
                # trajectory, coupling-matrix trajectory -- these can run
                # into the GB range per wavelength) before moving on.
                # This is the fix for OOM kernel crashes after a few
                # wavelengths: without it, all n_wavelengths engines stay
                # resident simultaneously.
                del engine, u0_batch, uf_batch
                _force_release_memory()

        return spectra_complex

    # ------------------------------------------------------------------
    @staticmethod
    def interpolate_complex_spectra(
        spectra_native: np.ndarray,
        wl_native_nm: np.ndarray,
        wl_output_nm: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate complex per-fiber coefficients from a coarse native
        wavelength grid onto a finer output grid.

        Amplitude and (unwrapped) phase are interpolated separately rather
        than interpolating real/imaginary parts directly -- linear
        interpolation of Re/Im would average incorrectly through any zero
        crossing that occurs between native samples. This is only valid
        where the coupling coefficients vary smoothly with wavelength;
        sample the native grid more densely near known mode-degeneracy /
        avoided-crossing regions, where amplitude and phase can change
        rapidly over a few nm and linear interpolation (of amplitude/phase
        or anything else) will not capture that structure.

        Parameters
        ----------
        spectra_native : ndarray, shape (n_wl_native, n_fields, n_fibers), complex
        wl_native_nm   : ndarray, shape (n_wl_native,)
        wl_output_nm   : ndarray, shape (n_wl_output,)

        Returns
        -------
        spectra_output : ndarray, shape (n_wl_output, n_fields, n_fibers), complex128
        """
        wl_native_nm = np.asarray(wl_native_nm, dtype=np.float64)
        wl_output_nm = np.asarray(wl_output_nm, dtype=np.float64)

        if wl_output_nm.min() < wl_native_nm.min() or wl_output_nm.max() > wl_native_nm.max():
            print("[MultiWavelength] WARNING: output wavelength grid extends "
                  "beyond the native solve range -- edge values will be held "
                  "flat by np.interp (extrapolation), not physically modeled.")

        n_wl_native, n_fields, n_fibers = spectra_native.shape
        n_wl_output = len(wl_output_nm)

        amp = np.abs(spectra_native)
        phase = np.unwrap(np.angle(spectra_native), axis=0)

        # Flatten (fields, fibers) into one axis for straightforward
        # per-column interpolation.
        amp_flat = amp.reshape(n_wl_native, -1)
        phase_flat = phase.reshape(n_wl_native, -1)

        amp_out = np.empty((n_wl_output, amp_flat.shape[1]))
        phase_out = np.empty((n_wl_output, amp_flat.shape[1]))
        for k in range(amp_flat.shape[1]):
            amp_out[:, k] = np.interp(wl_output_nm, wl_native_nm, amp_flat[:, k])
            phase_out[:, k] = np.interp(wl_output_nm, wl_native_nm, phase_flat[:, k])

        spectra_output = (amp_out * np.exp(1j * phase_out)).reshape(
            n_wl_output, n_fields, n_fibers)
        return spectra_output

    # ------------------------------------------------------------------
    def get_power_spectra(
        self,
        aberration_coeff_batch: np.ndarray,
        output_wavelengths_nm: Optional[np.ndarray] = None,
        use_gpu: Optional[bool] = None,
        gen_chunk_size: Optional[int] = None,
        prop_chunk_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper: propagate at all native wavelengths, optionally
        interpolate onto a finer output grid, and return real per-fiber
        power (flux) spectra ready for SpectralImageSimulator.

        The interpolation is done on the *complex* coefficients (amplitude
        and unwrapped phase separately, via interpolate_complex_spectra) and
        only then squared -- linearly interpolating already-squared power on
        a coarse grid smears out the nulls where |c|**2 -> 0.

        Returns
        -------
        power_spectra : ndarray, shape (n_wl_output, n_fields, n_fibers)
            np.abs(complex coefficient)**2 at each wavelength.
        wl_output_nm  : ndarray, shape (n_wl_output,)
            The wavelength grid the output is sampled on (== native grid
            if output_wavelengths_nm was not given).
        """
        spectra_complex = self.propagate_batch(
            aberration_coeff_batch, use_gpu=use_gpu,
            gen_chunk_size=gen_chunk_size, prop_chunk_size=prop_chunk_size,
        )

        if output_wavelengths_nm is None:
            return np.abs(spectra_complex) ** 2, self.native_wavelengths_nm

        wl_output_nm = np.asarray(output_wavelengths_nm, dtype=np.float64)
        spectra_out = self.interpolate_complex_spectra(
            spectra_complex, self.native_wavelengths_nm, wl_output_nm)
        return np.abs(spectra_out) ** 2, wl_output_nm

