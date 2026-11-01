"""Mode-bookkeeping diagnostics for the two-segment lantern ChainPropagator.

The bookkeeping itself lives in construct.py: the front segment (z = 0 ..
z_split) tracks every mode with the input-end degenerate pairs grouped, and
the back segment drops the modes that have become cladding modes by the
junction, read off the characterized front segment. This module only looks
at the n_eff spectra (untracked FEM solves), to check that picture at a
wavelength before spending the time to characterize it.
"""
from __future__ import annotations
import shutil
import tempfile
import numpy as np
from typing import List, Optional, Sequence

from cbeam.propagator import Propagator

from .params import scale_params_to_wavelength, build_lantern_geometry


def probe_neffs(
    wl_um: float,
    wvg,
    n_modes: int,
    zs: Sequence[float],
    verbose: bool = True,
) -> List[np.ndarray]:
    """
    One untracked FEM solve at each z in *zs* on a throwaway Propagator.
    Returns one descending-sorted length-``n_modes`` real n_eff array per z.
    """
    # throwaway save_dir: Propagator.__init__ makes ./data/<sub>/ folders,
    # which a probe has no business leaving behind.
    tmp = tempfile.mkdtemp(prefix="cbeam_probe_")
    try:
        probe = Propagator(wl_um, wvg, n_modes, save_dir=tmp)
        out = []
        for z in zs:
            if verbose:
                print(f"    [probe] solve_at(z={z:g}) ...")
            n, _ = probe.solve_at(float(z))
            out.append(np.sort(np.real(np.asarray(n, dtype=np.complex128)))[::-1])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out



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



def diagnose_mode_bookkeeping(
    base_params: dict,
    wavelength_nm: float,
    *,
    PL_N=None,
    n_modes: int = 20,
    z_split: Optional[float] = None,
    degen_atol: float = 1e-5,
) -> None:
    """
    Print the n_eff spectrum at the input and at the front/back junction
    for one wavelength, without characterizing anything.

    The input column marks the near-degenerate multiplets (gap below
    ``degen_atol``): the symmetry pairs there should match
    default_degenerate_groups_front; extra ones (e.g. modes 17-19 near the
    jacket cutoff from ~850 nm up) are accidental and are deliberately not
    grouped. The junction count is the number of modes still core-bound
    there (n_eff above the cladding index): it must be n_output_positions
    for the back segment to keep one mode per core.
    """
    p_lambda = scale_params_to_wavelength(base_params, wavelength_nm)
    PL_N     = PL_N if PL_N is not None else build_lantern_geometry(p_lambda)
    z_split  = p_lambda["z_ex"] / 2.0 if z_split is None else z_split
    n0, nj = probe_neffs(p_lambda["wl"], PL_N, n_modes, [0.0, z_split], verbose=False)

    nclad = p_lambda["nclad"]
    multi = [run for run in _multiplets(n0, degen_atol) if len(run) > 1]
    gid = {m: g for g, run in enumerate(multi) for m in run}
    print(f"    {wavelength_nm:.0f} nm    (n_eff - n_clad) x 1e4, sorted at each z")
    print("    mode |  input  junction | multiplet at input")
    for m in range(n_modes):
        print(f"    {m:4d} | {1e4 * (n0[m] - nclad):6.2f}  {1e4 * (nj[m] - nclad):7.2f} "
              f"| {gid.get(m, '.')}")
    print(f"    near-degenerate at the input: {multi}")
    print(f"    core-bound at the junction: {int(np.sum(nj > nclad))} "
          f"(cores: {base_params['n_output_positions']})")
