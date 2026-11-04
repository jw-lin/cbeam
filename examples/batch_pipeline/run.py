"""Entry point: the end-to-end batch propagation demo."""
from __future__ import annotations
import numpy as np

from .constants import backend
from .config import get_simulation_parameters
from .engine import build_and_characterize_lantern, BatchPropagationPipeline
from .plots import visualize_batch_output, batch_statistics


def main_batch_propagation(ifunc):
    """Main execution function for the batch propagation pipeline.

    Parameters
    ----------
    ifunc : the DM influence-function object, e.g.
        ``batch_pipeline.load_ifunc(get_simulation_parameters()["ifunc_file"])``
        -- a specula ``IFunc`` when specula is installed, a plain-numpy
        ``SimpleIFunc`` otherwise. See examples/BatchedPiston.ipynb.
    """
    print("=" * 60)
    print("PHOTONIC LANTERN BATCH PROPAGATION PIPELINE")
    print("=" * 60)

    p = get_simulation_parameters()
    print(f"\n[1/4] Configuration loaded.")
    print(f"      Wavelength: {p['wl']} μm  |  Lantern length: {p['z_ex']} μm")
    print(f"      Global backend: {backend}")
    print(f"      Influence function modes: {len(ifunc.influence_function)}")

    print(f"\n[2/4] Building waveguide propagator...")
    prop12 = build_and_characterize_lantern(p)
    print(f"      Modes: 20, Segments: 2")

    print(f"\n[3/4] Initialising batch pipeline ...")
    # field_gen_on_gpu=False  → field generation uses NumPy/SciPy (CPU)
    # field_gen_chunk_size=64 → process 64 fields at a time during field gen
    pipeline = BatchPropagationPipeline(
        prop12, p, ifunc,
        field_gen_on_gpu=False,
        field_gen_chunk_size=64,
    )

    mode_indices  = [0, 3, 5]
    amplitudes_nm = [0.0, 100.0, 300.0, 600.0]
    # One config per (mode, amplitude) pair: mode index -> column, amplitude -> value.
    mode_amp_pairs = [(m, a) for m in mode_indices for a in amplitudes_nm]
    print(f"      Batch size: {len(mode_amp_pairs)} fields")
    print(f"      Modes: {mode_indices}, Amplitudes (nm): {amplitudes_nm}")

    print(f"\n[4/4] Generating and propagating batch...")
    n_total      = pipeline._field_gen_np.num_modes
    coeff_matrix = np.zeros((len(mode_amp_pairs), n_total), dtype=np.float64)
    for k, (mode_idx, amp) in enumerate(mode_amp_pairs):
        coeff_matrix[k, mode_idx] = amp

    # chunk_size=None → use the instance default (field_gen_chunk_size=64).
    # Pass an explicit int to override, e.g. chunk_size=32 for tighter memory.
    u0_batch = pipeline.generate_batch_modal_coefficients(
        coeff_matrix, use_gpu=False)
    print(f"      Input field batch shape: {u0_batch.shape}")

    # chunk_size=8 is conservative for propagation; increase if GPU allows.
    uf_batch, zs, us_batch = pipeline.propagate_batch(u0_batch, chunk_size=8)
    print(f"      Output modal coefficient batch shape: {uf_batch.shape}")

    E_output_batch = pipeline.reconstruct_batch_output_fields(uf_batch)
    print(f"      Output spatial field batch shape: {E_output_batch.shape}")

    uf_2d_batch, X_plot, Y_plot = pipeline.interpolate_output_to_grid(E_output_batch)
    print(f"      Output grid batch shape: {uf_2d_batch.shape}")

    titles = [f"Mode {mode_idx}, {amp:.0f} nm"
              for mode_idx, amp in mode_amp_pairs]

    print(f"\n[Results] Visualising batch output...")
    visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles)
    batch_statistics(uf_batch, titles)

    print("\n" + "=" * 60)
    print("BATCH PROPAGATION COMPLETE")
    print("=" * 60)

    return {
        'u0_batch':       u0_batch,
        'uf_batch':       uf_batch,
        'E_output_batch': E_output_batch,
        'uf_2d_batch':    uf_2d_batch,
        'zs':             zs,
        'us_batch':       us_batch,
        'configs':        mode_amp_pairs,
        'pipeline':       pipeline,
    }


if __name__ == "__main__":
    from .ifunc import load_ifunc
    main_batch_propagation(load_ifunc(get_simulation_parameters()["ifunc_file"]))
