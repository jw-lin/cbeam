# Auto-split from the former single-file multi_wvl_pipeline.py.
"""Automatic per-wavelength degen_groups / skipped_modes from two endpoint FEM solves."""
from __future__ import annotations
import shutil
import tempfile
import numpy as np
from typing import List, Optional, Sequence, Tuple

from cbeam.propagator import Propagator

from .params import scale_params_to_wavelength, build_lantern_geometry


def _ior_index(wvg, *prefixes: str) -> Optional[float]:
    """Median of the IOR-dict values whose label starts with any of
    *prefixes* (case-insensitive). None if no label matches."""
    ior  = wvg.assign_IOR()
    vals = [float(v) for k, v in ior.items()
            if str(k).lower().startswith(tuple(p.lower() for p in prefixes))]
    return float(np.median(vals)) if vals else None



def _cladding_index(wvg) -> float:
    n = _ior_index(wvg, "clad")
    if n is None:
        raise ValueError(f"no 'clad*' entry in IOR dict; keys={list(wvg.assign_IOR())}")
    return n



def _cutoff_index(wvg, end: str) -> float:
    """Reference index below which a mode is unguided at a given lantern
    end. 'front' (multimode input): the jacket. 'back' (isolated output):
    the cladding. Falls back to the cladding if there is no jacket."""
    if end == "front":
        return _ior_index(wvg, "jack") or _cladding_index(wvg)
    return _cladding_index(wvg)



def probe_endpoint_neffs(
    wl_um: float,
    wvg,
    n_modes: int,
    z_ex: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One untracked FEM solve at each lantern end (z = 0 and z = z_ex) on a
    throwaway Propagator. Returns ``(neff0, neffL)``, each a
    descending-sorted length-``n_modes`` real array. This is all the
    automatic bookkeeping needs -- no adaptive sweep, no mode tracking.
    """
    z_ex = float(wvg.z_ex if z_ex is None else z_ex)
    # throwaway save_dir: Propagator.__init__ makes ./data/<sub>/ folders,
    # which a probe has no business leaving behind.
    tmp = tempfile.mkdtemp(prefix="cbeam_probe_")
    try:
        probe = Propagator(wl_um, wvg, n_modes, save_dir=tmp)
        if verbose:
            print(f"    [probe] solve_at(z=0) and solve_at(z={z_ex:g}) ...")
        n0, _ = probe.solve_at(0.0)
        nL, _ = probe.solve_at(z_ex)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n0 = np.sort(np.real(np.asarray(n0, dtype=np.complex128)))[::-1]
    nL = np.sort(np.real(np.asarray(nL, dtype=np.complex128)))[::-1]
    return n0, nL



def _multiplets(neff_desc: np.ndarray, atol: float) -> List[List[int]]:
    """Cut a descending-sorted n_eff spectrum into degenerate multiplets:
    a new multiplet starts wherever the gap to the previous mode exceeds
    ``atol``. Returns a list of index lists covering 0..len-1."""
    runs: List[List[int]] = [[0]]
    for i in range(1, len(neff_desc)):
        if neff_desc[i - 1] - neff_desc[i] <= atol:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs



def infer_lantern_mode_bookkeeping(
    wl_um: float,
    wvg,
    n_modes: int,
    *,
    n_expected_guided: Optional[int] = None,
    z_ex: Optional[float] = None,
    degen_atol: float = 1e-5,
    cutoff_margin: float = 5e-5,
    verbose: bool = True,
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """
    Derive ``(degen_groups_front, degen_groups_back, skipped_modes)`` for
    the lantern at one wavelength from two endpoint FEM solves.

    (Earlier versions took a ``z_split`` positional and an ``n_probe``
    keyword for a partial ``compute_neffs`` sweep; the current implementation
    uses two clean endpoint solves and needs neither.  ``diagnose_mode_
    bookkeeping`` still swallows those keywords for old notebook cells.)

    Parameters
    ----------
    wl_um : float
        Vacuum wavelength in micrometres (``p_lambda["wl"]``).
    wvg : PhotonicLantern
        The (wavelength-independent) geometry.
    n_modes : int
        Nmax tracked by the Propagator.
    n_expected_guided : int or None
        Number of genuinely guided modes (= number of output cores). When
        given, exactly ``n_modes - n_expected_guided`` modes are skipped
        (the lowest n_eff at the multimode input). When None, skips come
        from the jacket-proximity test at the input.
    degen_atol : float
        n_eff gap below which adjacent modes are one multiplet. ~1e-5
        catches true LP doublets (gap ~1e-8..1e-5) without merging
        distinct-but-close modes (gap ~1e-4).
    cutoff_margin : float
        Used only for the ``n_expected_guided=None`` skip path and for the
        "looks guided" warning.

    Returns
    -------
    degen_groups_front, degen_groups_back : list[list[int]]
    skipped_modes : list[int]

    Notes
    -----
    * Front groups are read at z=0 (multimode input) in eigensolver order,
      which is the index convention compute_modes() starts the front
      segment in.
    * Back groups are read at z=z_ex among the kept modes; for a lantern
      these collapse to a single near-degenerate block, which is
      invariant under the front->back index permutation.
    * A multiplet that contains the skipped mode AND a kept mode is still
      emitted as a front group (with the skipped index included) so
      correct_degeneracy() can lock the kept partner's rotation against
      the soon-to-be-zeroed spurious one -- this is what stabilises the
      band edge (e.g. modes 18-19 degenerate at 850-860 nm).
    """
    z_ex = float(wvg.z_ex if z_ex is None else z_ex)
    n0, nL = probe_endpoint_neffs(wl_um, wvg, n_modes, z_ex, verbose)

    n_cut_front = _cutoff_index(wvg, "front")
    n_cut_back  = _cutoff_index(wvg, "back")

    # ---- skipped modes: weakest-bound at the multimode input ----
    if n_expected_guided is not None:
        n_skip  = max(0, n_modes - int(n_expected_guided))
        skipped = sorted(int(i) for i in np.argsort(n0)[:n_skip])
        for i in skipped:
            if verbose and (n0[i] - n_cut_front) > cutoff_margin and \
               (nL[i] - n_cut_back) > cutoff_margin:
                print(f"    [warn] mode {i} skipped to meet "
                      f"n_expected_guided={n_expected_guided} but is above "
                      f"cutoff at both ends -- check the expected count.")
    else:
        skipped = sorted(int(i) for i in range(n_modes)
                         if (n0[i] - n_cut_front) <= cutoff_margin)
    skipped_set = set(skipped)

    # ---- front degenerate groups: multiplets at z=0 ----
    degen_front: List[List[int]] = []
    for run in _multiplets(n0, degen_atol):
        if len(run) < 2:
            continue
        if any(i not in skipped_set for i in run):   # has a kept member
            degen_front.append(sorted(run))

    # ---- back degenerate groups: multiplets at z=z_ex, kept modes only ----
    degen_back: List[List[int]] = []
    for run in _multiplets(nL, degen_atol):
        kept = [i for i in run if i not in skipped_set]
        if len(kept) >= 2:
            degen_back.append(sorted(kept))

    if verbose:
        _print_mode_bookkeeping_table(
            n0, nL, n_cut_front, n_cut_back, degen_front, degen_back, skipped)

    return degen_front, degen_back, skipped



def _print_mode_bookkeeping_table(
    n0: np.ndarray,
    nL: np.ndarray,
    n_cut_front: float,
    n_cut_back: float,
    degen_front: List[List[int]],
    degen_back: List[List[int]],
    skipped: Sequence[int],
) -> None:
    fg_id = {m: gid for gid, g in enumerate(degen_front) for m in g}
    bg_id = {m: gid for gid, g in enumerate(degen_back) for m in g}
    sk = set(skipped)
    print("    mode |  n_eff@0    n_eff@end | head_in   head_out | frontG backG | state")
    print("    -----+----------------------+--------------------+--------------+------")
    for m in range(len(n0)):
        hi, ho = n0[m] - n_cut_front, nL[m] - n_cut_back
        print(f"    {m:4d} | {n0[m]:9.6f}  {nL[m]:9.6f} | {hi:+.2e} {ho:+.2e} "
              f"| {str(fg_id.get(m,'.')):>5} {str(bg_id.get(m,'.')):>4} "
              f"| {'SKIP' if m in sk else 'guided'}")
    print(f"    cutoff index: front(jacket)={n_cut_front:.6f}  back(clad)={n_cut_back:.6f}")
    print(f"    degen_groups front = {degen_front}")
    print(f"    degen_groups back  = {degen_back}")
    print(f"    skipped_modes      = {sorted(skipped)}")



def diagnose_mode_bookkeeping(
    base_params: dict,
    wavelength_nm: float,
    *,
    PL_N: Optional[PhotonicLantern] = None,
    n_modes: int = 20,
    **infer_kwargs,
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """
    Notebook convenience: run infer_lantern_mode_bookkeeping() for one
    wavelength without characterising anything. Use it to inspect the
    inferred structure for every wavelength you plan to (re)characterise
    before committing the CPU time.
    """
    # absorb the old no-op knobs so existing notebook cells that still pass
    # `n_probe=` / `z_split=` keep working (see infer_lantern_mode_bookkeeping)
    infer_kwargs.pop("n_probe", None)
    infer_kwargs.pop("z_split", None)

    p_lambda = scale_params_to_wavelength(base_params, wavelength_nm)
    PL_N     = PL_N if PL_N is not None else build_lantern_geometry(p_lambda)
    return infer_lantern_mode_bookkeeping(
        p_lambda["wl"], PL_N, n_modes,
        n_expected_guided=base_params["n_output_positions"],
        z_ex=p_lambda["z_ex"], verbose=True, **infer_kwargs,
    )

