"""Per-wavelength ChainPropagator construction, with load-or-characterize caching."""
from __future__ import annotations
import os
import numpy as np
from typing import List, Optional, Tuple

from cbeam.propagator import Propagator, ChainPropagator
from cbeam.waveguide import PhotonicLantern

from batch_pipeline.constants import default_degenerate_groups_front
from batch_pipeline.engine import wavelength_tag

from .params import scale_params_to_wavelength, build_lantern_geometry


def _ensure_uint64_cells(mesh) -> None:
    if mesh is None:
        return
    for cellblock in getattr(mesh, "cells", []):
        if cellblock is None:
            continue  # skip any None cellblocks
        if cellblock.data.dtype != np.uint64:
            cellblock.data = cellblock.data.astype(np.uint64)


def _load_and_backfill_cmats(prop: Propagator, tag: str, verbose: bool = True) -> None:
    """
    Propagator.load(tag), then make sure the coupling matrices are cached
    too. cbeam's load() treats a missing cplcoeffs file as non-fatal (it
    prints a message and carries on without them), and computing them is
    ~10-30 s per segment, so if the file is missing compute them now and
    save them, reusing the zs / vs / mesh load() just restored -- later
    loads of this tag are then fast again.
    """
    prop.load(tag)
    if not os.path.exists(_cache_file(prop, "cplcoeffs", tag)):
        if verbose:
            print(f"    [{tag}] coupling-matrix cache missing -- computing "
                  f"and saving it now so future loads of this tag are fast ...")
        _ensure_uint64_cells(getattr(prop, "mesh", None))
        try:
            prop.compute_cmats(save=True, tag=tag)
        except (FileNotFoundError, OSError) as exc:
            # the caller treats FileNotFoundError/OSError as "no cache, go
            # characterize" -- a failure to *save* the backfilled cmats is
            # not that, so don't let it be misread as a cache miss.
            raise RuntimeError(
                f"coupling-matrix backfill for tag '{tag}' failed: {exc}") from exc
        if verbose:
            print(f"    [{tag}] coupling matrix computed and cached.")



def _load_or_characterize(
    prop: Propagator,
    tag: str,
    zi: float,
    zf: float,
    verbose: bool = True,
) -> None:
    """
    Load a Propagator's mode basis / coupling coefficients from cache if
    available (Propagator.load(tag)); otherwise solve for them from
    scratch with Propagator.characterize(zi, zf, save=True, tag=tag) and
    let cbeam cache the result under that tag for next time.

    characterize() is what actually does the FEM eigenmode solve at each
    z step and computes the coupling-coefficient matrices -- it's the
    same call that produces the "19port_0800_front-all"/"_back-clad"
    cache files of the single-wavelength pipeline. This is a genuine
    solve, not the diffrax-based coupled-mode propagation done later in
    .propagate() -- it's slow (~minutes) and is the real cost of adding a
    wavelength.

    Note: this catches (FileNotFoundError, OSError) as "no cache present".
    If your cbeam version raises something else for a missing tag (e.g. a
    bare Exception from a failed unpickle), widen the except clause
    accordingly -- the intent is "anything that means the file isn't
    there", not "silently swallow a real solve failure".
    """
    try:
        _load_and_backfill_cmats(prop, tag, verbose=verbose)
        if verbose:
            print(f"    [{tag}] loaded cached characterization.")
    except (FileNotFoundError, OSError) as exc:
        if verbose:
            print(f"    [{tag}] no cached characterization found ({exc}); "
                  f"running characterize(zi={zi}, zf={zf}, save=True) -- "
                  f"this can take several minutes ...")
        prop.characterize(zi, zf, save=True, tag=tag)
        if verbose:
            print(f"    [{tag}] characterization complete and cached.")


# On-disk layout written by cbeam's Propagator.characterize(save=True, tag=...)
# and read back by Propagator.load(tag): one "<sub>/<sub>_<tag>.npy" file per
# item, under the propagator's save_dir.  Mirrored here (rather than exposed by
# cbeam) so we can check whether one cached item exists without a load();
# keep in sync if a future cbeam version changes the layout.
_DEFAULT_SAVE_DIR = "./data"


def _cache_file(prop: Propagator, sub: str, tag: str) -> str:
    """Path of one cached item, e.g. <save_dir>/zvals/zvals_<tag>.npy."""
    base = getattr(prop, "save_dir", None) or _DEFAULT_SAVE_DIR
    return os.path.join(base, sub, f"{sub}_{tag}.npy")



def build_and_characterize_lantern_at_wavelength(
    base_params: dict,
    wavelength_nm: float,
    n_modes: int = 20,
    degen_groups: Optional[List[List[int]]] = None,
    z_split: Optional[float] = None,
    cache_prefix: str = "port",
    PL_N: Optional[PhotonicLantern] = None,
    verbose: bool = True,
) -> Tuple[dict, ChainPropagator]:
    """
    Build a ChainPropagator for the lantern at a single wavelength,
    characterizing it from scratch if no cached data exists yet.

    Mirrors build_and_characterize_lantern() from the single-wavelength
    pipeline, including its mode bookkeeping and cache tags (so at 800 nm
    both share one cache), but parameterises the wavelength:

    * the front segment tracks every mode and groups the input-end
      degenerate pairs (default_degenerate_groups_front: the same pairs
      from 700 to 900 nm). Modes that are only accidentally degenerate at
      the input -- e.g. 17-19 near the jacket cutoff from ~850 nm up -- are
      NOT grouped: a degen group carries one averaged n_eff along the whole
      segment, and these modes split up through the taper, some into
      cores, some into the cladding;
    * the back segment skips the modes that have become cladding modes by
      the junction (n_eff below the cladding index at the end of the front
      segment) and groups the rest -- one per output core.

    Parameters
    ----------
    base_params : dict
        Output of get_simulation_parameters(), used as a template. `wl`
        and `wavelength_nm` are overridden per call; geometry, taper,
        indices and mesh resolutions are assumed wavelength-independent.
    wavelength_nm : float
    n_modes : int
        Number of local modes tracked by the Propagator.
    degen_groups : list of lists, optional
        Explicit front-segment degen_groups, instead of
        default_degenerate_groups_front.
    z_split : float or None
        z coordinate separating the "front" and "back" Propagator
        segments. Defaults to base_params["z_ex"] / 2, matching the
        symmetric L1 == L2 == 50000 split used elsewhere in this codebase.
        Override if your front/back split isn't at the midpoint.
    cache_prefix : str
        Prefix for the load/save tag, e.g. str(n_output_positions) + "port" -> "19port_0800_front-all".
    PL_N : PhotonicLantern or None
        A pre-built, wavelength-independent lantern geometry to reuse
        (see build_lantern_geometry()). If None, one is built here --
        fine for a single call, wasteful if called once per wavelength in
        a loop. MultiWavelengthPropagationPipeline builds one PL_N and
        passes it to every wavelength.

    Returns
    -------
    p_lambda : dict
        Copy of base_params with wl / wavelength_nm set for this wavelength.
    prop12   : ChainPropagator
    """
    p_lambda = scale_params_to_wavelength(base_params, wavelength_nm)

    z_ex = p_lambda["z_ex"]
    _z_split = z_split if z_split is not None else z_ex / 2.0
    n_output_positions = str(base_params["n_output_positions"])

    tag = f"{n_output_positions}{cache_prefix}_{wavelength_tag(wavelength_nm)}"

    PL_N = PL_N if PL_N is not None else build_lantern_geometry(p_lambda)

    prop1 = Propagator(p_lambda["wl"], PL_N, n_modes)
    prop2 = Propagator(p_lambda["wl"], PL_N, n_modes)

    _front_tag, _back_tag = f"{tag}_front-all", f"{tag}_back-clad"

    if degen_groups is None:
        degen_groups = default_degenerate_groups_front[base_params["nrings"]]
    # copy so an in-place mutation by the propagator can't poison the
    # module global for the next wavelength in a loop
    prop1.degen_groups = [list(g) for g in degen_groups]
    prop1.skipped_modes = set()
    _load_or_characterize(prop1, _front_tag, 0.0, _z_split, verbose=verbose)

    n_junction = np.real(np.asarray(prop1.neffs[-1]))
    cladding = {k for k in range(n_modes) if n_junction[k] < p_lambda["nclad"]}
    if n_modes - len(cladding) != base_params["n_output_positions"]:
        raise RuntimeError(
            f"[{tag}] {n_modes - len(cladding)} tracked modes are core-bound at "
            f"the junction (z={_z_split:g}), expected one per output core "
            f"({base_params['n_output_positions']}); n_eff - n_clad there: "
            f"{np.round(n_junction - p_lambda['nclad'], 6).tolist()}. Track more "
            f"modes (n_modes) or check the front segment's degen_groups (a group"
            f" that mixes core- and cladding-bound modes cannot be split here).")
    if verbose:
        print(f"    [{tag}] cladding mode(s) at the junction, skipped in the "
              f"back segment: {sorted(cladding)}")

    prop2.skipped_modes = cladding
    prop2.degen_groups = [[k for k in range(n_modes) if k not in cladding]]
    # Must happen before load/characterize: the back segment's initial
    # eigenbasis is bootstrapped from the front segment's final one,
    # whether the front segment was just loaded or just solved.
    prop2.load_init_conds(prop1)
    _load_or_characterize(prop2, _back_tag, _z_split, z_ex, verbose=verbose)

    return p_lambda, ChainPropagator([prop1, prop2])
