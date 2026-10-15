# =====================================================================
# MULTI-WAVELENGTH RESULTS VISUALIZATION
# =====================================================================
# Plotting helpers for the outputs of MultiWavelengthPropagationPipeline
# (power_spectra, wl_nm) and, optionally, the downstream detector-plane
# stage (SpectralImageSimulator / SpectralExtractor from
# spectral_extraction_module.py).
#
# All functions take plain arrays (power_spectra, wl_nm, amplitudes) --
# nothing here depends on pipeline internals, so they work equally well
# fed by a live run or by arrays you've saved to disk.
# =====================================================================

from contextlib import contextmanager
from typing import Optional, Sequence, List

import numpy as np
import matplotlib.pyplot as plt


@contextmanager
def _figure(ax=None, **subplots_kw):
    """Yield ``(fig, ax_or_axes)``.

    If ``ax`` is given, draw onto it and leave the figure to the caller;
    otherwise make a fresh figure with ``plt.subplots(**subplots_kw)`` and
    ``fig.tight_layout()`` + ``plt.show()`` it on exit.  Removes the repeated
    own-figure / tight_layout / show boilerplate from every plot function.
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(**subplots_kw)
    else:
        fig = ax.figure
    try:
        yield fig, ax
    finally:
        if own:
            fig.tight_layout()
            plt.show()


# =====================================================================
# SINGLE-FIELD SPECTRA
# =====================================================================

def plot_fiber_spectra(
    power_spectra: np.ndarray,
    wl_nm: np.ndarray,
    field_idx: int,
    fiber_indices: Optional[Sequence[int]] = None,
    ax=None,
    title: Optional[str] = None,
    plot_sum: bool = True,
):
    """
    Overlay per-fiber power spectra for one field on a single axes.

    Parameters
    ----------
    power_spectra : ndarray, shape (n_wl, n_fields, n_fibers)
    wl_nm         : ndarray, shape (n_wl,)
    field_idx     : int
    fiber_indices : sequence of int, optional
        Which fibers to plot. Defaults to all fibers (can get busy for
        19 -- use plot_spectra_grid() for a clearer all-fiber view).
    ax : matplotlib Axes, optional
        Draw onto this axes; if None a new figure is made and shown.
    title : str, optional
        Axes title; defaults to "Field <field_idx> -- per-fiber spectra".
    plot_sum : bool, default True
        Also overlay the all-cores total as a dashed black line.
    """
    n_fibers = power_spectra.shape[2]
    if fiber_indices is None:
        fiber_indices = range(n_fibers)

    with _figure(ax, figsize=(9, 5)) as (fig, ax):
        cmap = plt.cm.viridis
        for k, fiber_idx in enumerate(fiber_indices):
            color = cmap(k / max(1, len(fiber_indices) - 1))
            ax.plot(wl_nm, power_spectra[:, field_idx, fiber_idx],
                    label=f"Fiber {fiber_idx}", color=color, linewidth=1.2)

        if plot_sum:
            total = power_spectra[:, field_idx, :].sum(axis=1)
            ax.plot(wl_nm, total, label="Sum (all cores)",
                    color="black", linewidth=1.8, linestyle="--")

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Power (a.u.)")
        ax.set_title(title or f"Field {field_idx} -- per-fiber spectra")
        if len(fiber_indices) <= 8:
            ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    return ax


def plot_spectra_grid(
    power_spectra: np.ndarray,
    wl_nm: np.ndarray,
    field_idx: int,
    title: Optional[str] = None,
):
    """
    Plot every fiber's spectrum for one field in its own small subplot
    (4x5 grid, matching the 19-fiber layout), so you can see all of them
    without an overcrowded single-axes overlay.
    """
    n_fibers = power_spectra.shape[2]
    n_cols = 5
    n_rows = int(np.ceil(n_fibers / n_cols))

    with _figure(nrows=n_rows, ncols=n_cols,
                 figsize=(4 * n_cols, 2.5 * n_rows), sharex=True) as (fig, axes):
        axes = np.atleast_1d(axes).flatten()

        for fiber_idx in range(n_fibers):
            ax = axes[fiber_idx]
            ax.plot(wl_nm, power_spectra[:, field_idx, fiber_idx], color="tab:blue", linewidth=0.9)
            ax.set_title(f"Fiber {fiber_idx}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

        for ax in axes[n_fibers:]:
            ax.set_visible(False)

        fig.supxlabel("Wavelength (nm)")
        fig.supylabel("Power (a.u.)")
        fig.suptitle(title or f"Field {field_idx} -- all fiber spectra", fontweight="bold")


def plot_wavelength_fiber_heatmap(
    power_spectra: np.ndarray,
    wl_nm: np.ndarray,
    field_idx: int,
    ax=None,
    title: Optional[str] = None,
    log_scale: bool = False,
):
    """
    Heatmap of power(wavelength, fiber) for one field -- a quick way to
    see which fibers carry flux and how that shifts with wavelength.
    """
    n_fibers = power_spectra.shape[2]
    data = power_spectra[:, field_idx, :]  # (n_wl, n_fibers)
    if log_scale:
        # relative floor (power is in a.u.): clamp at 12 decades below the peak
        floor = 1e-12 * max(float(data.max()), 1.0)
        data = np.log10(np.maximum(data, floor))

    with _figure(ax, figsize=(8, 6)) as (fig, ax):
        im = ax.imshow(
            data, aspect="auto", origin="lower", cmap="inferno",
            extent=[0, n_fibers, wl_nm.min(), wl_nm.max()],
        )
        ax.set_xlabel("Fiber index")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_title(title or f"Field {field_idx} -- power(wavelength, fiber)"
                              f"{' [log10]' if log_scale else ''}")
        fig.colorbar(im, ax=ax, label="log10(Power)" if log_scale else "Power (a.u.)")
    return ax


# =====================================================================
# ABERRATION-RAMP COMPARISONS
# =====================================================================

def plot_ramp_response(
    power_spectra: np.ndarray,
    wl_nm: np.ndarray,
    amplitudes: np.ndarray,
    fiber_idx: int,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    """
    Overlay one fiber's spectrum across a ramp of aberration amplitudes
    (e.g. from create_ramp_aberration_configs), color-coded by amplitude.

    Parameters
    ----------
    power_spectra : ndarray, shape (n_wl, n_fields, n_fibers)
    wl_nm         : ndarray, shape (n_wl,)
    amplitudes    : ndarray, shape (n_fields,)
        The scalar aberration amplitude for each field -- e.g.
        coeff_matrix[:, mode_idx] if you ramped a single mode.
    fiber_idx     : int
    labels        : list of str, optional
        Per-field labels (e.g. from create_ramp_aberration_configs) used
        for the legend instead of raw amplitude values.
    """
    n_fields = power_spectra.shape[1]
    amplitudes = np.asarray(amplitudes)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(amplitudes.min(), amplitudes.max())

    with _figure(figsize=(9, 5)) as (fig, ax):
        for field_idx in range(n_fields):
            color = cmap(norm(amplitudes[field_idx]))
            ax.plot(wl_nm, power_spectra[:, field_idx, fiber_idx],
                    color=color, linewidth=1.3,
                    label=(labels[field_idx] if labels is not None
                           else f"{amplitudes[field_idx]:.1f}"))

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Power (a.u.)")
        ax.set_title(title or f"Fiber {fiber_idx} -- response across aberration ramp")
        ax.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Aberration amplitude")


def plot_total_throughput_vs_amplitude(
    power_spectra: np.ndarray,
    amplitudes: np.ndarray,
    labels: Optional[List[str]] = None,
    mode_label: Optional[str] = None,
):
    """
    Total power (summed over wavelength AND fiber) vs aberration
    amplitude -- the simplest possible diagnostic of how much light is
    lost from the lantern as the aberration grows.
    """
    total_power = power_spectra.sum(axis=(0, 2))  # (n_fields,)
    amplitudes = np.asarray(amplitudes)
    order = np.argsort(amplitudes)

    with _figure(figsize=(7, 5)) as (fig, ax):
        ax.plot(amplitudes[order], total_power[order], "o-", color="tab:red")
        ax.set_xlabel(f"Aberration amplitude{f' ({mode_label})' if mode_label else ''}")
        ax.set_ylabel("Total power, summed over wavelength & fiber (a.u.)")
        ax.set_title("Total lantern throughput vs. aberration amplitude")
        ax.grid(True, alpha=0.3)


def plot_per_fiber_throughput_vs_amplitude(
    power_spectra: np.ndarray,
    amplitudes: np.ndarray,
    fiber_indices: Optional[Sequence[int]] = None,
    mode_label: Optional[str] = None,
):
    """
    Per-fiber power (integrated over wavelength) vs aberration amplitude
    -- shows which fibers gain/lose light as the aberration grows, i.e.
    the modal-coupling signature of that aberration mode.
    """
    n_fibers = power_spectra.shape[2]
    if fiber_indices is None:
        fiber_indices = range(n_fibers)

    amplitudes = np.asarray(amplitudes)
    order = np.argsort(amplitudes)
    power_per_fiber = power_spectra.sum(axis=0)  # (n_fields, n_fibers)

    with _figure(figsize=(9, 5)) as (fig, ax):
        cmap = plt.cm.viridis
        for k, fiber_idx in enumerate(fiber_indices):
            color = cmap(k / max(1, len(fiber_indices) - 1))
            ax.plot(amplitudes[order], power_per_fiber[order, fiber_idx],
                    "o-", color=color, markersize=3, linewidth=1.1,
                    label=f"Fiber {fiber_idx}")

        ax.set_xlabel(f"Aberration amplitude{f' ({mode_label})' if mode_label else ''}")
        ax.set_ylabel("Power, integrated over wavelength (a.u.)")
        ax.set_title("Per-fiber throughput vs. aberration amplitude")
        ax.grid(True, alpha=0.3)
        if len(fiber_indices) <= 8:
            ax.legend(fontsize=8, ncol=2)


# =====================================================================
# DETECTOR-STAGE CONVENIENCE WRAPPER
# =====================================================================

def plot_detector_stage_for_field(
    power_spectra: np.ndarray,
    wl_nm: np.ndarray,
    field_idx: int,
    fiber_indices: Optional[Sequence[int]] = None,
):
    """
    Run one field's spectra through SpectralImageSimulator /
    SpectralExtractor (from the ``spectral`` package) and plot the
    simulated detector image alongside the extracted spectra -- the same
    two visualize_* calls used in ``spectral.run``'s own example, wrapped
    so you can call it directly on multi-wavelength pipeline output
    without re-deriving spectra_field0 / spectral_cfg each time.

    ``multi_wvl_pipeline`` is imported inside the function on purpose:
    every *other* function in this module needs only numpy + matplotlib and
    works on saved arrays, so importing the (cbeam / juliacall / specula)
    stack is deferred to this one call.  The ``spectral`` imports are light
    (numpy/scipy/matplotlib) and sit here just to keep them by their use.
    """
    from spectral.simulate import SpectralImageSimulator
    from spectral.extract import SpectralExtractor
    from spectral.plots import visualize_spectral_image, visualize_extracted_spectra
    from multi_wvl_pipeline import (
        spectra_for_field, build_spectral_config_from_wavelength_grid,
    )

    spectral_cfg = build_spectral_config_from_wavelength_grid(wl_nm)
    simulator = SpectralImageSimulator(spectral_cfg)
    extractor = SpectralExtractor(spectral_cfg)

    spectra_field = spectra_for_field(power_spectra, field_idx=field_idx)
    image, fiber_positions = simulator.create_spectral_image(
        modal_powers=None, spectra=spectra_field, add_noise=True, exposure_time=1.0,
    )
    extracted, variance = extractor.extract_spectra(image, fiber_positions, method="optimal")

    visualize_spectral_image(image, fiber_positions, spectral_cfg,
                              title=f"Field {field_idx} -- simulated detector image")

    n_fibers = power_spectra.shape[2]
    if fiber_indices is None:
        fiber_indices = [0, n_fibers // 4, n_fibers // 2, 3 * n_fibers // 4, n_fibers - 1]
    visualize_extracted_spectra(
        extracted, wl_nm, variance, fiber_indices=list(fiber_indices),
        title=f"Field {field_idx} -- extracted spectra",
    )

    return image, extracted, variance