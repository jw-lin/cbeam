# Auto-split from the former single-file multi_wvl_pipeline.py.
"""End-to-end multi-wavelength propagation demo."""
from __future__ import annotations
import numpy as np

from .pipeline import MultiWavelengthPropagationPipeline
from .glue import spectra_for_field, build_spectral_config_from_wavelength_grid


def example_multiwavelength_propagation():
    """
    End-to-end example: propagate a batch of aberration configurations at
    several native wavelengths, interpolate onto a fine grid, and hand the
    result to the spectral extraction module.
    """
    import specula
    specula.init(0)
    from specula.data_objects.ifunc import IFunc
    from batch_pipeline.config import get_simulation_parameters, create_random_aberration_configs
    from spectral.simulate import SpectralImageSimulator
    from spectral.extract import SpectralExtractor

    print("=" * 60)
    print("MULTI-WAVELENGTH PHOTONIC LANTERN PROPAGATION")
    print("=" * 60)

    base_params = get_simulation_parameters()
    ifunc = IFunc.restore(base_params["ifunc_file"])

    # Coarse native solve grid: physically propagate at these wavelengths.
    # Any wavelength without a cached characterization will be solved and
    # cached automatically on this call -- with 21 points and no existing
    # cache, expect on the order of a few hours the first time this runs
    # (see the COST NOTE at the top of this file / the class docstring).
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

    # Fine output grid for the dispersed spectra (interpolated from the
    # coarse native grid above).
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

    # Feed field 0's spectra into the detector-plane simulation.
    spectral_cfg = build_spectral_config_from_wavelength_grid(wl_out)
    simulator = SpectralImageSimulator(spectral_cfg)
    extractor = SpectralExtractor(spectral_cfg)

    spectra_field0 = spectra_for_field(power_spectra, field_idx=0)
    image, fiber_positions = simulator.create_spectral_image(
        modal_powers=None, spectra=spectra_field0, add_noise=True, exposure_time=1.0,
    )
    extracted, variance = extractor.extract_spectra(image, fiber_positions, method='optimal')

    print(f"Extracted spectra shape: {extracted.shape}")
    return power_spectra, wl_out, image, extracted


if __name__ == "__main__":
    example_multiwavelength_propagation()

if __name__ == "__main__":
    example_multiwavelength_propagation()
