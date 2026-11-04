"""DM influence-function loading, with or without specula.

batch_pipeline only ever reads two attributes of an influence-function
object: ``mask_inf_func`` (the 2-D pupil mask) and ``influence_function``
(the ``(n_modes, n_pupil_pixels)`` mode matrix, one value per non-zero mask
pixel in row-major order). specula's ``IFunc`` provides them, possibly as
CuPy arrays on a GPU; ``SimpleIFunc`` provides the same two arrays as plain
numpy, read straight from the same FITS file.

Both paths hand the pipeline bit-identical arrays -- ``SimpleIFunc`` uses the
dtypes specula itself ends up with at its default precision (float64 mode
matrix, float32 mask) -- so every downstream result is the same whether or
not specula is installed.
"""
from __future__ import annotations
import numpy as np


def to_numpy(a):
    """Host numpy copy of a numpy or CuPy array (CuPy arrays expose ``.get()``)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


class SimpleIFunc:
    """Minimal numpy stand-in for ``specula.data_objects.ifunc.IFunc``."""

    def __init__(self, influence_function, mask_inf_func):
        self.influence_function = np.asarray(influence_function, dtype=np.float64)
        self.mask_inf_func      = np.asarray(mask_inf_func, dtype=np.float32)
        n_pix = int(np.count_nonzero(self.mask_inf_func))
        if self.influence_function.ndim != 2 or self.influence_function.shape[1] != n_pix:
            raise ValueError(
                f"influence_function has shape {self.influence_function.shape}; "
                f"expected (n_modes, {n_pix}) for a mask with {n_pix} pupil pixels")

    @staticmethod
    def restore(filename, exten=1):
        """Read the FITS layout ``specula.data_objects.ifunc.IFunc.save`` writes:
        HDU ``exten`` = mode matrix stored as (n_pixels, n_modes), HDU
        ``exten + 1`` = mask."""
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise ImportError(
                "reading an influence-function FITS file without specula needs "
                "astropy (pip install astropy)") from exc
        with fits.open(filename) as hdul:
            ifunc = hdul[exten].data.T
            mask  = hdul[exten + 1].data
        return SimpleIFunc(ifunc, mask)


def load_ifunc(filename, use_specula=None):
    """Load a DM influence function from a specula-format FITS file.

    use_specula : None (default) uses specula when it is importable and
        ``SimpleIFunc`` otherwise; True / False forces one or the other.
        When specula is used but has not been initialised yet, it is
        initialised on the CPU (``specula.init(-1)``); call ``specula.init``
        yourself beforehand to pick a GPU instead.
    """
    if use_specula is None:
        try:
            import specula  # noqa: F401
            use_specula = True
        except ImportError:
            use_specula = False
    if not use_specula:
        return SimpleIFunc.restore(filename)

    import specula
    if specula.xp is None:
        specula.init(-1)
    from specula.data_objects.ifunc import IFunc
    return IFunc.restore(filename)


def center_ifunc_mask(ifunc):
    """Shift ``ifunc``'s pupil mask, in place, so its centroid lands on the
    grid centre (diagnose_input_psf flags an off-centre mask as a PSF
    alignment problem). Returns the applied ``(shift_y, shift_x)`` in pixels.

    Only the mask moves: ``influence_function`` holds one value per mask
    pixel in row-major order, and a translation that wraps no pixel across
    the grid edge keeps that order, so the mode matrix stays valid as is.
    """
    mask = to_numpy(ifunc.mask_inf_func)
    rows, cols = np.where(mask > 0)
    shift_y = int(mask.shape[0] // 2 - np.mean(rows))
    shift_x = int(mask.shape[1] // 2 - np.mean(cols))
    if not (0 <= rows.min() + shift_y and rows.max() + shift_y < mask.shape[0]
            and 0 <= cols.min() + shift_x and cols.max() + shift_x < mask.shape[1]):
        raise ValueError(
            f"centering shift ({shift_y}, {shift_x}) would wrap pupil pixels "
            f"across the edge of the {mask.shape} grid")

    centered = np.roll(np.roll(mask, shift_y, axis=0), shift_x, axis=1)
    if isinstance(ifunc, SimpleIFunc):
        ifunc.mask_inf_func = centered.astype(np.float32)
    else:
        # specula IFunc: the property setter moves the array to its device
        # and refreshes the cached pupil-pixel indices.
        ifunc.mask_inf_func = centered
    return shift_y, shift_x
