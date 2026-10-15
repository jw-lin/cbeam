"""Per-wavelength ChainPropagator construction, with load-or-characterize caching."""
from __future__ import annotations
import os
import numpy as np
from typing import List, Optional, Tuple

from cbeam.propagator import Propagator, ChainPropagator
from cbeam.waveguide import PhotonicLantern

from batch_pipeline.constants import (
    default_degenerate_groups_front, default_degenerate_groups_back,
    default_skipped_modes_front, default_skipped_modes_back,
)
from batch_pipeline.engine import wavelength_tag

from .params import scale_params_to_wavelength, build_lantern_geometry
from .bookkeeping import infer_lantern_mode_bookkeeping


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
    same call that produced the original "19port_0800_front"/"_back"
    cache files (see cbeam's 19-port lantern example). This is a genuine
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
# cbeam) so we can cheaply test "is this tag already cached?" without a load();
# keep in sync if a future cbeam version changes the layout.
_CACHE_SUBDIRS = ("eigenvalues", "eigenmodes", "zvals")
_DEFAULT_SAVE_DIR = "./data"


def _cache_file(prop: Propagator, sub: str, tag: str) -> str:
    """Path of one cached item, e.g. <save_dir>/zvals/zvals_<tag>.npy."""
    base = getattr(prop, "save_dir", None) or _DEFAULT_SAVE_DIR
    return os.path.join(base, sub, f"{sub}_{tag}.npy")


def _characterization_cached(prop: Propagator, tag: str) -> bool:
    """True iff the files Propagator.load(tag) needs are already on disk."""
    return all(os.path.exists(_cache_file(prop, sub, tag)) for sub in _CACHE_SUBDIRS)



def build_and_characterize_lantern_at_wavelength(
    base_params: dict,
    wavelength_nm: float,
    n_modes: int = 20,
    skipped_modes: Optional[List[int]] = None,
    degen_groups: Optional[List[List[int]]] = None,
    z_split: Optional[float] = None,
    cache_prefix: str = "port",
    PL_N: Optional[PhotonicLantern] = None,
    auto_mode_bookkeeping: bool = False,
    verbose: bool = True,
) -> Tuple[dict, ChainPropagator]:
    """
    Build a ChainPropagator for the lantern at a single wavelength,
    characterizing it from scratch if no cached data exists yet.

    Mirrors build_and_characterize_lantern() from the single-wavelength
    pipeline, but parameterises the wavelength and the mode-cache filename
    tag instead of hard-coding "0800", and produces the cache instead of
    assuming it's already there.

    Parameters
    ----------
    base_params : dict
        Output of get_simulation_parameters(), used as a template. `wl`
        and `wavelength_nm` are overridden per call; geometry, taper,
        indices and mesh resolutions are assumed wavelength-independent.
    wavelength_nm : float
    n_modes : int
        Number of local modes tracked by the Propagator.
    skipped_modes, degen_groups : optional overrides
        Explicit overrides win over everything. Default reproduces the
        single-wavelength pipeline's configuration ([18] skipped; the
        front-half degenerate LP-mode pairs grouped) -- valid at 800 nm,
        NOT across wavelength (LP degeneracies split, avoided crossings
        move as V changes). For any wavelength away from 800 nm, either
        pass explicit lists or set auto_mode_bookkeeping=True.
    auto_mode_bookkeeping : bool
        When True and neither skipped_modes nor degen_groups is given,
        and this wavelength is about to be characterised from scratch
        (no cache), derive degen_groups (front + back separately) and
        skipped_modes automatically from FEM solves at the two lantern
        ends -- see infer_lantern_mode_bookkeeping(). No effect when the cache is
        already present (the probe would be wasted).
    z_split : float or None
        z coordinate separating the "front" and "back" Propagator
        segments. Defaults to base_params["z_ex"] / 2, matching the
        symmetric L1 == L2 == 50000 split used elsewhere in this codebase.
        Override if your front/back split isn't at the midpoint.
    cache_prefix : str
        Prefix for the load/save tag, e.g. str(n_output_positions) + "port" -> "19port_0800_front".
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

    _front_tag, _back_tag = f"{tag}_front", f"{tag}_back"
    _explicit = (skipped_modes is not None) or (degen_groups is not None)
    _will_solve = not (_characterization_cached(prop1, _front_tag)
                       and _characterization_cached(prop2, _back_tag))

    if auto_mode_bookkeeping and not _explicit and _will_solve:
        if verbose:
            print(f"    [{tag}] auto_mode_bookkeeping: probing n_eff(z) to derive "
                  f"degen_groups / skipped_modes for this wavelength ...")
        _degen_front, _degen_back, _skipped = infer_lantern_mode_bookkeeping(
            p_lambda["wl"], PL_N, n_modes,
            n_expected_guided=base_params["n_output_positions"],
            z_ex=z_ex, verbose=verbose,
        )
        _skipped_set = set(_skipped)
    else:
        _skipped = list(skipped_modes) if skipped_modes is not None else [18]
        _skipped_set = set(_skipped)
        # Copy the module-global grouping so an in-place mutation by the
        # propagator can't poison it for the next wavelength in a loop
        # (matches build_and_characterize_lantern in the single-wl pipeline).
        _degen_front = (
            [list(g) for g in degen_groups] if degen_groups is not None
            else [list(g) for g in default_degenerate_groups_front[base_params["nrings"]]]
        )
        _degen_back = [[i for i in range(n_modes) if i not in _skipped_set]]

    prop1.degen_groups = _degen_front
    prop1.skipped_modes = _skipped
    _load_or_characterize(prop1, _front_tag, 0.0, _z_split, verbose=verbose)

    prop2.skipped_modes = _skipped
    prop2.degen_groups = _degen_back
    # Must happen before load/characterize: the back segment's initial
    # eigenbasis is bootstrapped from the front segment's final one,
    # whether the front segment was just loaded or just solved.
    prop2.load_init_conds(prop1)
    _load_or_characterize(prop2, _back_tag, _z_split, z_ex, verbose=verbose)

    return p_lambda, ChainPropagator([prop1, prop2])

