# Split out of the former single-file spectral_extraction_module.py.
"""All matplotlib output for the spectral pipeline (the only module that imports it)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from .config import SpectralConfig


def visualize_spectral_image(
    image: np.ndarray,
    fiber_positions: np.ndarray,
    config: SpectralConfig,
    title: str = "Simulated Spectral Image",
):
    """2D spectral image (log) with the fiber traces marked, plus a zoom."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    im1 = ax1.imshow(np.log10(image + 1), aspect="auto", cmap="viridis", origin="lower")
    ax1.set_xlabel("Wavelength (pixel)")
    ax1.set_ylabel("Spatial (pixel)")
    ax1.set_title(f"{title} (log scale)")
    for i, pos in enumerate(fiber_positions):
        ax1.axhline(pos, color="red", alpha=0.3, linewidth=0.5)
        ax1.text(10, pos, f"{i}", color="red", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.colorbar(im1, ax=ax1, label="log10(Counts + 1)")

    y_mid = config.detector_height // 2
    x_mid = config.detector_width // 2
    zoom = 100
    im2 = ax2.imshow(
        image[y_mid - zoom:y_mid + zoom, x_mid - zoom:x_mid + zoom],
        aspect="auto", cmap="viridis", origin="lower",
        extent=[x_mid - zoom, x_mid + zoom, y_mid - zoom, y_mid + zoom],
    )
    ax2.set_xlabel("Wavelength (pixel)")
    ax2.set_ylabel("Spatial (pixel)")
    ax2.set_title("Zoomed Region")
    plt.colorbar(im2, ax=ax2, label="Counts")

    plt.tight_layout()
    plt.show()


def visualize_extracted_spectra(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    variance: Optional[np.ndarray] = None,
    fiber_indices: Optional[List[int]] = None,
    title: str = "Extracted Spectra",
):
    """Stacked flux-vs-wavelength panels for a handful of fibers."""
    if fiber_indices is None:
        fiber_indices = list(range(min(5, spectra.shape[1])))

    fig, axes = plt.subplots(len(fiber_indices), 1,
                             figsize=(12, 3 * len(fiber_indices)))
    if len(fiber_indices) == 1:
        axes = [axes]

    for ax, fiber_idx in zip(axes, fiber_indices):
        spec = spectra[:, fiber_idx]
        ax.plot(wavelengths, spec, "k-", linewidth=1, label=f"Fiber {fiber_idx}")
        if variance is not None:
            sigma = np.sqrt(variance[:, fiber_idx])
            ax.fill_between(wavelengths, spec - sigma, spec + sigma,
                            alpha=0.3, label="+/-1 sigma")
        ax.set_ylabel("Flux (counts)")
        ax.set_title(f"{title} - Fiber {fiber_idx}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Wavelength (nm)")
    plt.tight_layout()
    plt.show()


def visualize_all_spectra_grid(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    title: Optional[str] = None,
):
    """Every fiber spectrum in a near-square grid of small panels."""
    n_fibers = spectra.shape[1]
    n_cols = int(np.ceil(np.sqrt(n_fibers)))
    n_rows = int(np.ceil(n_fibers / n_cols))
    if title is None:
        title = f"All {n_fibers} Fiber Spectra"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for fiber_idx in range(n_fibers):
        ax = axes[fiber_idx]
        ax.plot(wavelengths, spectra[:, fiber_idx], "b-", linewidth=0.8)
        ax.set_title(f"Fiber {fiber_idx}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if fiber_idx >= n_fibers - n_cols:
            ax.set_xlabel("lambda (nm)", fontsize=9)
        if fiber_idx % n_cols == 0:
            ax.set_ylabel("Flux", fontsize=9)

    for j in range(n_fibers, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
