# Examples

Run these from inside `examples/` (they import `batch_pipeline` from here and
write their characterization cache to `examples/data/`, which is git-ignored).

## Batched propagation

`batch_pipeline/` propagates whole batches of input fields through a
characterized 19-port photonic lantern in a single `propagate()` call, on the
numpy or the jax backend (`CBEAM_BACKEND`).

| | |
|---|---|
| `BatchedMinimal.ipynb` | start here: cbeam alone, on the 6-port lantern of the docs -- one `propagate()` call for a batch of fields vs. a loop, and output-core powers for the whole batch |
| `Batched.ipynb` | timing of batched vs. one-field-at-a-time `propagate()`, numpy and jax |
| `BatchedPiston.ipynb` | the full pipeline: DM-aberrated input pupils -> batched propagation -> output fields -> per-core signals |
| `python -m batch_pipeline.run` | the same pipeline as a script |
| `ifunc/` | the DM influence function the pipeline uses (see its README) |

Besides cbeam's own requirements:

- **jax + diffrax** for the jax backend (`pip install -e ".[jax]"`);
- **astropy** to read the influence-function FITS file;
- **specula** is optional. When it is installed the influence function is
  loaded through it, otherwise through plain numpy; the results are
  bit-identical either way.

The first run characterizes the lantern (several minutes per wavelength);
later runs load the cached characterization from `examples/data/`.

## Multi-wavelength propagation

`multi_wvl/` propagates a batch of aberrated fields at every wavelength of a
native grid -- one lantern characterization per wavelength, built and released
one at a time -- and returns each output fiber's spectrum as the fraction of
the incident power reaching it.

| | |
|---|---|
| `multi_wavelength.ipynb` | 41 wavelengths (700-900 nm) x a petal-piston ramp: output spectra, input coupling vs. wavelength |
| `multi_wavelength_vis.py` | the notebook's plotting helpers |
| `python -m multi_wvl.run` | a shorter end-to-end run as a script |

Each wavelength needs its own characterization: minutes per wavelength the
first time, cached in `examples/data/` afterwards (41 wavelengths take a few
hours to characterize).
