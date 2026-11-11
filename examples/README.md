# Examples

Run these from inside `examples/` (they import `batch_pipeline` from here and
write their characterization cache to `examples/data/`, which is git-ignored).

## Batched propagation

`batch_pipeline/` propagates whole batches of input fields through a
characterized waveguide in a single `propagate()` call, on the numpy or the jax
backend (`CBEAM_BACKEND`). `BatchPropagationPipeline` takes any characterized
`Propagator` or `ChainPropagator`; the input fields are DM-aberrated telescope
pupils imaged onto its input face. The notebooks use a 19-port photonic
lantern.

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
