"""End-to-end multi-wavelength propagation demo."""
from __future__ import annotations
import numpy as np

from .pipeline import MultiWavelengthPropagationPipeline


def example_multiwavelength_propagation():
    """
    End-to-end example: propagate a batch of aberration configurations at
    several native wavelengths and interpolate the output-fiber spectra
    onto a fine grid.
    """
    from batch_pipeline import (
        get_simulation_parameters, create_random_aberration_configs,
        load_ifunc, center_ifunc_mask,
    )

    print("=" * 60)
    print("MULTI-WAVELENGTH PHOTONIC LANTERN PROPAGATION")
    print("=" * 60)

    base_params = get_simulation_parameters()
    ifunc = load_ifunc(base_params["ifunc_file"])   # specula if installed
    center_ifunc_mask(ifunc)

    # Coarse native solve grid: physically propagate at these wavelengths.
    # Any wavelength without a cached characterization will be solved and
    # cached automatically on this call -- with 21 points and no existing
    # cache, expect on the order of a few hours the first time this runs
    # (see "Notes on cost" in MultiWavelengthPropagationPipeline).
    native_wl_nm = np.linspace(700.0, 900.0, 21)

    mwp = MultiWavelengthPropagationPipeline(
        base_params=base_params,
        ifunc=ifunc,
        native_wavelengths_nm=native_wl_nm,
        field_gen_on_gpu=False,
        field_gen_chunk_size=64,
    )

    n_total_modes = mwp.num_modes
    coeff_matrix, _labels = create_random_aberration_configs(
        n=8, m=n_total_modes, minv=-100.0, maxv=100.0)

    # Fine output grid for the spectra (interpolated from the coarse native
    # grid above).
    output_wl_nm = np.linspace(700.0, 900.0, 200)

    power_spectra, wl_out = mwp.get_power_spectra(
        coeff_matrix,
        output_wavelengths_nm=output_wl_nm,
        use_gpu=False,
        prop_chunk_size=8,
    )
    print(f"\npower_spectra shape: {power_spectra.shape}  "
          f"(n_wavelengths={power_spectra.shape[0]}, "
          f"n_fields={power_spectra.shape[1]}, n_fibers={power_spectra.shape[2]})")
    return power_spectra, wl_out


if __name__ == "__main__":
    example_multiwavelength_propagation()
