"""All matplotlib output for the batch pipeline (the only module that imports it)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter, map_coordinates

from .constants import N_SIGNALS, DEFAULT_GRID_RESOLUTION
from .analysis import (
    calibrate_subpixel_centers, collect_subpixel_signals,
    map_evaluated_to_ideal_geometry,
)


def diagnose_input_psf(pipeline):
    """Diagnostic for the input PSF: mask centring, FFT-grid centring, mesh
    interpolation and coupling efficiency, with a 2x3 summary figure."""
    print("\n=== RUNNING INPUT PSF CENTERING & SCALE DIAGNOSTIC ===")

    # 1. Generate unaberrated field (zero coefficients)
    fg_np = pipeline._field_gen_np
    n_total_modes = fg_np.num_modes
    zero_coeffs = np.zeros((1, n_total_modes))

    # 2. Check mask centering BEFORE field generation
    mask = fg_np.mask_np
    mask_center_y = np.mean(np.where(mask > 0)[0])
    mask_center_x = np.mean(np.where(mask > 0)[1])
    # apply_ef_to_lantern's ifftshift puts pupil pixel N//2 on the optical
    # axis, so that is the centre the mask has to sit on.
    expected_center = mask.shape[0] // 2
    # center_ifunc_mask moves the mask by whole pixels, so a sub-pixel
    # residual is expected; only a residual of a pixel or more is flagged.
    mask_tol_px = 1.0
    
    print(f"\n[MASK CENTERING CHECK]")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask centroid: ({mask_center_y:.2f}, {mask_center_x:.2f})")
    print(f"  Expected center: ({expected_center:.2f}, {expected_center:.2f})")
    print(f"  Offset: ({mask_center_y - expected_center:.3f}, {mask_center_x - expected_center:.3f}) pixels")
    
    if abs(mask_center_y - expected_center) >= mask_tol_px or abs(mask_center_x - expected_center) >= mask_tol_px:
        print("  ⚠️  WARNING: Mask is off-center by a pixel or more! This will cause PSF misalignment.")
        print("  Recommendation: re-center it with batch_pipeline.center_ifunc_mask(ifunc) before building the pipeline.")

    # 3. Extract field at key stages
    Ef_input = fg_np.generate_field_profiles_batch(zero_coeffs)
    Ef_focal_grid = fg_np.apply_ef_to_lantern(Ef_input)[0]
    Ef_focal_mesh = fg_np.resample_to_mesh(Ef_focal_grid[None, ...])[0]

    intensity_grid = np.abs(Ef_focal_grid) ** 2
    intensity_mesh = np.abs(Ef_focal_mesh) ** 2
    
    # 4. Check FFT grid centering
    padded_size = fg_np._pad_geometry()[1]

    # FFT centre index (even vs odd padded grid)
    if padded_size % 2 == 0:
        # For even grids, FFT center is at index padded_size // 2
        fft_center_idx = padded_size // 2
    else:
        # For odd grids, FFT center is at index (padded_size - 1) // 2
        fft_center_idx = (padded_size - 1) // 2
    
    peak_idx = np.unravel_index(np.argmax(intensity_grid), intensity_grid.shape)
    offset_px = (peak_idx[0] - fft_center_idx, peak_idx[1] - fft_center_idx)
    
    print(f"\n[FFT GRID CHECK]")
    print(f"  FFT grid shape: {intensity_grid.shape}")
    print(f"  FFT center index: ({fft_center_idx}, {fft_center_idx})")
    print(f"  PSF peak index: {peak_idx}")
    print(f"  Centering offset: {offset_px} pixels")
    
    if abs(offset_px[0]) > 0.5 or abs(offset_px[1]) > 0.5:
        print(f"  ⚠️  WARNING: PSF peak is off-center by > 0.5 pixels")
        print(f"  This indicates a centering problem in the FFT or mask.")

    # 5. Centroid calculation (subpixel accuracy)
    total_intensity = intensity_grid.sum()
    y_centroid = np.sum(np.indices(intensity_grid.shape)[0] * intensity_grid) / total_intensity
    x_centroid = np.sum(np.indices(intensity_grid.shape)[1] * intensity_grid) / total_intensity
    centroid_offset = (y_centroid - fft_center_idx, x_centroid - fft_center_idx)
    
    print(f"  PSF centroid (subpixel): ({y_centroid:.3f}, {x_centroid:.3f})")
    print(f"  Centroid offset: ({centroid_offset[0]:.3f}, {centroid_offset[1]:.3f}) pixels")

    # 6. Check mesh interpolation
    mesh_pts = pipeline.wvg_props_input['mesh'].points
    x_min, x_max = mesh_pts[:, 0].min(), mesh_pts[:, 0].max()
    y_min, y_max = mesh_pts[:, 1].min(), mesh_pts[:, 1].max()
    # lantern cores at the input end (z=0, where the taper scale is 1)
    core_pts = np.asarray(pipeline.p["output_positions"], dtype=float)

    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 400),
                                 np.linspace(y_min, y_max, 400))
    mesh_grid_intensity = griddata((mesh_pts[:, 0], mesh_pts[:, 1]), intensity_mesh,
                                   (grid_x, grid_y), method='cubic', fill_value=0)

    peak_mesh_idx = np.unravel_index(np.argmax(mesh_grid_intensity), mesh_grid_intensity.shape)
    peak_mesh_x = grid_x[0, peak_mesh_idx[1]]
    peak_mesh_y = grid_y[peak_mesh_idx[0], 0]

    dist_to_cores = np.sqrt((core_pts[:, 0] - peak_mesh_x)**2 + (core_pts[:, 1] - peak_mesh_y)**2)
    min_dist = dist_to_cores.min()
    closest_core = np.argmin(dist_to_cores)
    
    print(f"\n[MESH INTERPOLATION CHECK]")
    print(f"  Mesh PSF peak: ({peak_mesh_x:.3f}, {peak_mesh_y:.3f}) μm")
    print(f"  Nearest core (#{closest_core}): distance = {min_dist:.3f} μm")
    
    if min_dist > 2.0:
        print(f"  ⚠️  WARNING: PSF peak is > 2 μm from nearest core")
        print(f"  This suggests a scaling or centering issue in mesh interpolation.")

    # 7. Power interception: power on the mesh as the FE quadratic form
    # E^H B E (a plain sum over nodes would weight them by mesh density),
    # and the share of it inside the multimode cladding at the input end.
    B = pipeline.wvg_props_input['B']
    rclad = pipeline.p["rclad"]
    inside = np.hypot(mesh_pts[:, 0], mesh_pts[:, 1]) < rclad
    E_in = np.where(inside, Ef_focal_mesh, 0)
    total_power = np.real(np.vdot(Ef_focal_mesh, B @ Ef_focal_mesh))
    power_in_clad = np.real(np.vdot(E_in, B @ E_in))
    frac_in_clad = power_in_clad / total_power if total_power > 0 else 0

    # fraction of the focal-plane power that couples into the tracked modes
    _, eff = pipeline._modal_proj_np.project_batch(Ef_focal_mesh[None, :])
    coupled_power = float(np.asarray(eff)[0] / fg_np.focal_plane_norm) ** 2

    print(f"\n[COUPLING EFFICIENCY CHECK]")
    print(f"  Power inside the cladding (r < {rclad} μm): {frac_in_clad:.1%} of the power on the mesh")
    print(f"  Power coupled into the tracked modes: {coupled_power:.1%} of the focal-plane power")

    if frac_in_clad < 0.5:
        print(f"  ⚠️  WARNING: < 50% of the PSF power falls inside the cladding")
        print(f"  Check beam size, alignment, and numerical aperture matching.")

    # 8. Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: FFT grid with corrected center lines
    ax = axes[0, 0]
    im = ax.imshow(np.log10(intensity_grid + 1e-8), cmap='viridis',
                   extent=[-fft_center_idx, padded_size - fft_center_idx,
                           -fft_center_idx, padded_size - fft_center_idx])
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Center')
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.plot(offset_px[1], offset_px[0], 'r+', markersize=15, markeredgewidth=2, label='Peak')
    ax.set_title(f"FFT Focal Plane (log)\nOffset: {offset_px}")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    ax.legend()
    plt.colorbar(im, ax=ax, label="log10 Intensity")

    # Plot 2: Mask
    ax = axes[0, 1]
    im = ax.imshow(mask, cmap='gray', origin='lower')
    ax.axvline(expected_center, color='r', linestyle='--', alpha=0.5)
    ax.axhline(expected_center, color='r', linestyle='--', alpha=0.5)
    ax.plot(mask_center_x, mask_center_y, 'rx', markersize=10, markeredgewidth=2)
    ax.set_title("Pupil Mask\nRed cross = centroid")
    plt.colorbar(im, ax=ax)

    # Plot 3: Cross-section
    ax = axes[0, 2]
    x_axis = np.arange(padded_size) - fft_center_idx
    y_peak = intensity_grid[peak_idx[0], :]
    ax.plot(x_axis, y_peak / y_peak.max(), label='Horizontal')
    x_peak = intensity_grid[:, peak_idx[1]]
    ax.plot(x_axis, x_peak / x_peak.max(), label='Vertical')
    ax.axvline(0, color='r', linestyle='--', alpha=0.3)
    ax.set_title("Normalized PSF Cross-sections")
    ax.set_xlabel("Pixels from center")
    ax.set_ylabel("Normalized intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: PSF on mesh
    ax = axes[1, 0]
    im = ax.imshow(mesh_grid_intensity, cmap='inferno',
                   extent=[x_min, x_max, y_min, y_max], origin='lower')
    ax.scatter(core_pts[:, 0], core_pts[:, 1], c='cyan', s=30, 
               edgecolor='k', linewidth=1.5, label='Cores')
    ax.scatter(peak_mesh_x, peak_mesh_y, c='red', s=80, marker='x',
               linewidth=2, label='PSF peak')
    ax.set_title(f"PSF on Mesh\nNearest core: {min_dist:.2f} μm")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.legend()
    plt.colorbar(im, ax=ax, label="Intensity")

    # Plot 5: Radial profile
    ax = axes[1, 1]
    r = np.sqrt((mesh_pts[:, 0] - peak_mesh_x)**2 + (mesh_pts[:, 1] - peak_mesh_y)**2)
    idx_sort = np.argsort(r)
    r_sorted = r[idx_sort]
    I_sorted = intensity_mesh[idx_sort]
    bins = np.linspace(0, r_sorted.max(), 50)
    r_bin = (bins[1:] + bins[:-1]) / 2
    I_bin = np.zeros_like(r_bin)
    for i in range(len(bins) - 1):
        mask_bin = (r_sorted >= bins[i]) & (r_sorted < bins[i+1])
        if np.any(mask_bin):
            I_bin[i] = I_sorted[mask_bin].mean()
    ax.plot(r_bin, I_bin, 'b-', linewidth=2, label='Measured')
    ax.set_title("Radial PSF Profile")
    ax.set_xlabel("Radius (μm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Pupil phase
    ax = axes[1, 2]
    phase_input = np.angle(Ef_input[0])
    im = ax.imshow(phase_input, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax.set_title("Pupil Phase")
    plt.colorbar(im, ax=ax, label="Phase (rad)")

    plt.tight_layout()
    plt.show()

    # Summary report
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    issues = []
    if abs(mask_center_y - expected_center) >= mask_tol_px or abs(mask_center_x - expected_center) >= mask_tol_px:
        issues.append("⚠️  Mask is off-center")
    if abs(offset_px[0]) > 0.5 or abs(offset_px[1]) > 0.5:
        issues.append("⚠️  FFT PSF peak is off-center")
    if min_dist > 2.0:
        issues.append("⚠️  Mesh PSF far from cores")
    if frac_in_clad < 0.5:
        issues.append("⚠️  Low PSF power inside the cladding")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nRecommendations:")
        print("  1. Check mask centering (batch_pipeline.center_ifunc_mask)")
        print("  2. Verify pixel_scale_um matches FFT grid spacing")
        print("  3. Check mesh interpolation coordinate mapping")
    else:
        print("✓ All checks passed - PSF appears well-centered")
    print("="*60)



def visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles=None, max_fields=1,
                           peak_box_size=5, show_arrow=True, *, maxv=None):
    """
    Visualise a batch of output intensity maps with peak detection and annotation.

    Parameters
    ----------
    uf_2d_batch : ndarray, shape (n_fields, nx, ny)
        Batch of intensity maps on the regular grid.
    X_plot, Y_plot : ndarray, shape (nx, ny) or (ny, nx)
        Meshgrid arrays of spatial coordinates (μm).
    titles : list of str, optional
        Titles for each field.
    max_fields : int
        Maximum number of fields to plot.
    peak_box_size : int (odd)
        Size of the square (in pixels) used for averaging around the peak.
    show_arrow : bool
        If True, draw a cyan arrow from the top‑right corner pointing to the peak.
    maxv : int, optional
        Deprecated alias for ``max_fields`` (the old name read like a value cap).
    """
    if maxv is not None:
        max_fields = maxv
    n_fields = min(uf_2d_batch.shape[0], max_fields)
    n_cols = min(3, n_fields)
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_fields == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_fields):
        ax = axes[i]
        # already real intensity |E|**2 from interpolate_output_to_grid()
        data_intensity = uf_2d_batch[i]

        # Log intensity for colormap (log10, matching diagnose_input_psf)
        im_log = np.log10(data_intensity + 1e-10)
        im = ax.imshow(im_log, cmap='inferno',
                       extent=[X_plot.min(), X_plot.max(),
                               Y_plot.min(), Y_plot.max()],
                       origin='lower')
        ax.set_title(titles[i] if titles else f"Field {i}")
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        plt.colorbar(im, ax=ax, label='log10(Intensity)')

        # ------- Peak detection as center of box with highest average signal -------
        # Compute average intensity in each box using convolution
        box_averages = uniform_filter(data_intensity, size=peak_box_size, 
                                      mode='constant', cval=0)
        
        # Find the box center with highest average
        peak_idx = np.unravel_index(np.argmax(box_averages), box_averages.shape)
        row, col = peak_idx
        
        # Physical coordinates
        x_peak = X_plot[row, col]
        y_peak = Y_plot[row, col]

        # Get statistics for that box
        half = peak_box_size // 2
        r0 = max(0, row - half)
        r1 = min(data_intensity.shape[0], row + half + 1)
        c0 = max(0, col - half)
        c1 = min(data_intensity.shape[1], col + half + 1)
        peak_region = data_intensity[r0:r1, c0:c1]
        peak_avg = np.mean(peak_region)
        peak_max = np.max(peak_region)

        # ------- Annotations -------
        # Red 'x' marker at the peak
        ax.plot(x_peak, y_peak, 'rx', markersize=12, markeredgewidth=2,
                label='Peak')

        # Text box with peak and average values
        ax.text(x_peak, y_peak,
                f'Peak: {peak_max:.2e}\nAvg({peak_box_size}x{peak_box_size}): {peak_avg:.2e}',
                color='white', fontsize=8, va='bottom', ha='left',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

        # Optional arrow pointing to the peak
        if show_arrow:
            # Arrow starting near the top‑right corner of the plot
            x_range = X_plot.max() - X_plot.min()
            y_range = Y_plot.max() - Y_plot.min()
            start_x = X_plot.max() - 0.05 * x_range
            start_y = Y_plot.max() - 0.05 * y_range
            arrow = FancyArrowPatch((start_x, start_y), (x_peak, y_peak),
                                    arrowstyle='->', mutation_scale=20,
                                    color='cyan', linewidth=2)
            ax.add_patch(arrow)

        # Legend (shows only 'Peak' entry)
        ax.legend(loc='upper right', fontsize=8)

    # Hide any unused subplots
    for j in range(n_fields, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def display_hex_grid_plots(ideal_centers, standardized_signals):
    """Generates the concurrent side-by-side 3D column and 2D map views."""
    heights = standardized_signals * 1000
    centers = np.array(ideal_centers)
    hex_radius = 0.4
    
    vmin, vmax = heights.min(), heights.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    
    min_x, max_x = centers[:, 0].min() - 1, centers[:, 0].max() + 1
    min_y, max_y = centers[:, 1].min() - 1, centers[:, 1].max() + 1
    
    def create_hexagon_vertices(cx, cy, r):
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        return np.column_stack((cx + r * np.cos(angles), cy + r * np.sin(angles)))

    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    n_sides = 6
    polys, colors_list = [], []
    for i, (x, y) in enumerate(centers):
        height = heights[i]
        color = cmap(norm(height))
        hex_vertices = create_hexagon_vertices(x, y, hex_radius)
        
        base_3d = np.column_stack((hex_vertices, np.zeros(n_sides)))
        top_3d = np.column_stack((hex_vertices, np.full(n_sides, height)))
        vertices_3d = np.vstack((base_3d, top_3d))
        
        polys.extend([vertices_3d[:n_sides], vertices_3d[n_sides:]])
        colors_list.extend([color, color])
        
        for j in range(n_sides):
            side_face = [vertices_3d[j], vertices_3d[(j+1)%n_sides], vertices_3d[(j+1)%n_sides + n_sides], vertices_3d[j + n_sides]]
            polys.append(side_face)
            colors_list.append(color)

    poly_collection = Poly3DCollection(polys, alpha=0.7, edgecolor='black', linewidth=0.5)
    poly_collection.set_facecolor(colors_list)
    ax1.add_collection3d(poly_collection)
    ax1.set_xlim(min_x, max_x); ax1.set_ylim(min_y, max_y); ax1.set_zlim(0, heights.max() * 1.1)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Height')
    ax1.set_title('3D Height Field on Hexagonal Grid')
    
    for i, (x, y) in enumerate(centers):
        height = heights[i]
        hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=0,
                                 facecolor=cmap(norm(height)), edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.add_patch(hexagon)
        ax2.text(x, y, f'{height:.1f}', ha='center', va='center', fontsize=8, fontweight='bold')

    ax2.set_xlim(min_x, max_x); ax2.set_ylim(min_y, max_y); ax2.set_aspect('equal')
    ax2.set_xlabel('X', fontsize=12); ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Top View: Hexagonal Grid Height Field', fontsize=14)
    ax2.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], pad=0.05, shrink=0.7).set_label('Height', fontsize=12)
    plt.show()


def visualize_batch_hex_grid_signals(
    pipeline,
    E_output_batch,
    ideal_grid_positions,
    titles=None,
    grid_resolution=DEFAULT_GRID_RESOLUTION,
):
    """Execute centroid sub-pixel tracking and display hex grid configurations.

    Uses the analysis helpers (calibrate_subpixel_centers,
    collect_subpixel_signals, map_evaluated_to_ideal_geometry) and the local
    display_hex_grid_plots directly -- they used to be passed in as arguments,
    but every caller passed the same four package functions.
    """
    mesh_final            = pipeline.wvg_props_output['mesh']
    waveguide_modes_final = pipeline.wvg_props_output['modes']

    # =====================================================================
    # Collapse (n_modes, n_mesh_points) into a 1D map (n_mesh_points,)
    # =====================================================================
    total_modes_profile = np.sum(np.abs(waveguide_modes_final) ** 2, axis=0)

    # Pass the 1D combined intensity profile instead of the raw 2D modes matrix
    core_centers, _, _, dx, dy = calibrate_subpixel_centers(
        total_modes_profile, mesh_final)
    # =====================================================================

    ideal_permutation = map_evaluated_to_ideal_geometry(
        core_centers, ideal_grid_positions)

    n_fields = E_output_batch.shape[0]
    print("\n" + "=" * 60)
    print("GEOMETRIC HEX SIGNAL EXTRACTION PROCESSING")
    print("=" * 60)

    uf_2d_batch, X_plot_out, Y_plot_out = pipeline.interpolate_output_to_grid(
        E_output_batch, grid_resolution)

    # Physical -> pixel mapping for the output grid, taken from the grid itself
    # (the core centres live in the same physical frame, over the mesh extent).
    x_min = X_plot_out[0, 0]
    y_min = Y_plot_out[0, 0]
    dx    = X_plot_out[0, 1] - X_plot_out[0, 0]
    dy    = Y_plot_out[1, 0] - Y_plot_out[0, 0]

    for i in range(n_fields):
        field_title = titles[i] if titles else f"Field {i}"
        print(f"Processing core integration tracks for: {field_title}...")

        output_signals = collect_subpixel_signals(
            uf_2d_batch[i], x_min, y_min, dx, dy, core_centers)

        standardized_signals = np.zeros(len(output_signals))
        standardized_signals[ideal_permutation] = output_signals

        print(f"Displaying Core Matrix for: {field_title}")
        display_hex_grid_plots(ideal_grid_positions, standardized_signals)


def batch_statistics(uf_batch, labels=None):
    """Compute and display statistics of batch propagation outputs."""
    n_fields = uf_batch.shape[0]
    print("\n" + "=" * 60)
    print("BATCH PROPAGATION STATISTICS")
    print("=" * 60)
    for i in range(n_fields):
        label       = labels[i] if labels else f"Field {i}"
        mode_powers = np.abs(uf_batch[i]) ** 2
        total_power = np.sum(mode_powers)
        max_mode    = np.argmax(mode_powers)
        top_3_power = np.sum(np.sort(mode_powers)[-3:])
        print(f"\n{label}:")
        print(f"  Total Power:          {total_power:.4f}")
        print(f"  Max Mode (idx):       {max_mode}")
        print(f"  Max Mode Power:       {mode_powers[max_mode]:.4f}")
        print(f"  Power in Top 3 Modes: {top_3_power:.4f}")
