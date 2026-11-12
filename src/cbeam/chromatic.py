""" characterize a waveguide over a range of wavelengths.

a ``Propagator`` is built for one wavelength: its eigenmodes, effective indices
and coupling coefficients all depend on it, and ``Propagator.wl`` is read-only.
the chromatic response of a device therefore takes one characterization per
wavelength. ``sweep()`` runs that loop for any waveguide: you supply a function
that returns a characterized ``Propagator`` (or ``ChainPropagator``) for a given
wavelength, and ``sweep()`` evaluates a set of measures on each one -- by default
the transfer matrix and the effective indices at both ends -- and stacks the
results along a leading wavelength axis. only one wavelength's propagator is
alive at a time.

example::

    from cbeam import waveguide, propagator, chromatic

    def build(wl):
        wvg = waveguide.Dicoupler(rcore, rcore, ncore, ncore, dmax, dmin, nclad,
                                  coupling_length, bend_length, core_res,
                                  core_mesh_size=1.0, clad_mesh_size=20.0)
        prop = propagator.Propagator(wl, wvg, Nmax=2)
        chromatic.load_or_characterize(prop, chromatic.wavelength_tag("dc", wl))
        return prop

    res = chromatic.sweep(build, np.linspace(1.45, 1.65, 21))
    power = np.abs(res["transfer_matrix"]) ** 2     # (n_wl, n_out, n_in)

the refractive indices can follow the wavelength too: build the waveguide with
indices from ``dispersive_index()`` (or any function of ``wl``) inside ``build``.
"""

from typing import Callable, Dict, Iterable, Optional

import numpy as np


def silica_index(wl):
    """ refractive index of fused silica (Malitson 1965 Sellmeier fit).

    ARGS:
        wl: wavelength in um (valid ~0.21-3.7 um)
    RETURNS:
        (float or array): the refractive index
    """
    l2 = np.asarray(wl, dtype=float) ** 2
    n2 = (1.0
          + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
          + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
          + 0.8974794 * l2 / (l2 - 9.896161 ** 2))
    return np.sqrt(n2)


def dispersive_index(n_ref, wl, wl_ref, material: Callable = silica_index):
    """ refractive index at ``wl`` of a material that has index ``n_ref`` at ``wl_ref``,
    shifted by the dispersion of a host material: n_ref + material(wl) - material(wl_ref).

    for doped-silica waveguides this keeps every index contrast (core-cladding,
    cladding-jacket, ...) fixed while the common level follows silica, a first-order
    chromatic model; pass each region's design index through it.

    ARGS:
        n_ref: index at the reference wavelength (float or array)
        wl: wavelength in um
        wl_ref: reference wavelength in um
        material: the host material's index as a function of wavelength in um
    RETURNS:
        (float or array): the index at ``wl``
    """
    return np.asarray(n_ref, dtype=float) + material(wl) - material(wl_ref)


def wavelength_tag(tag: str, wl: float) -> str:
    """ cache tag for one wavelength, e.g. ("dc", 1.55) -> "dc_1550.0nm".

    the tag is all ``Propagator.load()`` checks, so it must also encode anything
    else the characterization depends on (geometry, mesh, index model, ...).

    ARGS:
        tag: base tag
        wl: wavelength in um
    RETURNS:
        (str): ``tag`` with the wavelength in nm (0.1 nm resolution) appended
    """
    return f"{tag}_{wl * 1e3:.1f}nm"


def load_or_characterize(prop, tag: str, zi=None, zf=None) -> bool:
    """ load a saved characterization of ``prop`` under ``tag``, or characterize it
    from ``zi`` to ``zf`` and save it under ``tag``.

    ARGS:
        prop: a ``Propagator``, with any mode bookkeeping (``degen_groups``,
            ``skipped_modes``, initial conditions) already set
        tag: the save tag
        zi, zf: passed to ``Propagator.characterize()`` if no saved data exists
    RETURNS:
        (bool): True if ``prop`` was characterized, False if it was loaded
    """
    try:
        prop.load(tag)
        return False
    except (FileNotFoundError, OSError):
        prop.characterize(zi, zf, save=True, tag=tag)
        return True


def transfer_matrix(prop, channel_basis: bool = True) -> np.ndarray:
    """ the transfer matrix of a characterized ``Propagator`` or ``ChainPropagator``:
    column j is the output for unit input in mode j (zero for skipped modes).

    ARGS:
        prop: a characterized ``Propagator`` or ``ChainPropagator``
        channel_basis: express the output in the waveguide's channel basis (this
            needs a waveguide with output channels, e.g. ``PhotonicLantern`` or
            ``Dicoupler``); if False, in the eigenbasis at the output end
    RETURNS:
        (array): an Nmax x Nmax complex transfer matrix
    """
    if hasattr(prop, "compute_transfer_matrix"):
        return np.asarray(prop.compute_transfer_matrix(channel_basis=channel_basis))
    # ChainPropagator: same construction, column by column
    N = prop.Nmax
    skipped = prop.propagators[0].skipped_modes
    M = np.zeros((N, N), dtype=np.complex128)
    for j in range(N):
        if j in skipped:
            continue
        _, _, uf = prop.propagate(prop.xp.eye(N, dtype=prop.xp.complex128)[j])
        out = np.asarray(prop.to_channel_basis(uf) if channel_basis else uf)
        M[:len(out), j] = out
    return M


def neffs_in(prop) -> np.ndarray:
    """ effective indices of the modes at the input end of a characterized
    ``Propagator`` or ``ChainPropagator``. """
    first = prop.propagators[0] if hasattr(prop, "propagators") else prop
    return np.real(np.asarray(first.neffs[0]))


def neffs_out(prop) -> np.ndarray:
    """ effective indices of the modes at the output end of a characterized
    ``Propagator`` or ``ChainPropagator``. """
    last = prop.propagators[-1] if hasattr(prop, "propagators") else prop
    return np.real(np.asarray(last.neffs[-1]))


DEFAULT_MEASURES = {
    "transfer_matrix": transfer_matrix,
    "neffs_in": neffs_in,
    "neffs_out": neffs_out,
}


def sweep(build: Callable, wavelengths: Iterable[float],
          measures: Optional[Dict[str, Callable]] = None,
          verbose: bool = True) -> Dict[str, np.ndarray]:
    """ evaluate a waveguide's response at each of a set of wavelengths.

    ARGS:
        build: function ``build(wl)`` returning a characterized ``Propagator`` or
            ``ChainPropagator`` at wavelength ``wl`` (um). this is where the waveguide,
            its (possibly wavelength-dependent) indices and the mode bookkeeping are
            set up; use ``load_or_characterize()`` with ``wavelength_tag()`` to cache
            the characterizations.
        wavelengths: the wavelengths, in um
        measures: dict ``{name: f}``; each ``f(prop)`` returns an array of the same shape
            at every wavelength. defaults to ``DEFAULT_MEASURES`` (transfer matrix in the
            channel basis, effective indices at the input and output ends).
        verbose: print a line per wavelength
    RETURNS:
        (dict): ``{name: array}`` with each measure stacked along a leading wavelength
        axis, plus ``"wavelengths"``.
    """
    measures = DEFAULT_MEASURES if measures is None else measures
    wls = np.asarray(list(wavelengths), dtype=float)
    results = {name: [] for name in measures}
    for i, wl in enumerate(wls):
        if verbose:
            print(f"[chromatic] {i + 1}/{len(wls)}: wl = {wl:g} um", flush=True)
        prop = build(float(wl))
        for name, f in measures.items():
            results[name].append(np.asarray(f(prop)))
        del prop        # release this wavelength's modes before building the next
    out = {name: np.stack(vals) for name, vals in results.items()}
    out["wavelengths"] = wls
    return out
