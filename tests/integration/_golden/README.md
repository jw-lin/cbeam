# Integration-test reference data ("golden files")

Each `<module>__<test>.npz` holds the numerical outputs that the matching
integration test produced on a **known-good** run, keyed by a short name.  On a
normal test run the `golden` fixture (see `tests/conftest.py`) reloads these and
compares them with `numpy.testing.assert_allclose`, on top of the physical
invariant assertions the tests already make.

## Provenance

Generated from the **`tests`** branch library
(`git describe` = `e383525`, itself `9fc45db` + the test suite) with:

```
CBEAM_GOLDEN=record pytest tests/integration
```

This branch is the pre-refactor code path.  It is the agreed reference: a run on
any other branch is expected to reproduce these numbers.

## What is pinned

| quantity                              | tolerance (abs) | why that loose |
|---------------------------------------|-----------------|----------------|
| `neffs` effective-index arrays        | round-off       | eigenvalues are gauge-free |
| `neffs(z)` scans                      | round-off*      | resampled onto the run's own z-grid first |
| sorted channel-power spectra          | 1e-3            | which symmetric core carries which power can permute |
| mode-amplitude magnitudes             | 1e-3            | eigenvector sign is arbitrary |

`*` the 19-port `compute_neffs` scan is pinned at `rtol=1e-4` because its core
insertion order feeds the mesh triangulation; see
`waveguide.get_19port_positions`.

Per-mode power *within* a large degenerate subspace (the 19-port
`ChainPropagator`) is **not** pinned — it is gauge-dependent and moves by ~5e-2
between eigensolver bases.  Only the physical channel-power spectrum is checked
there.

## Regenerating

Only when the reference itself should move (a deliberate algorithm change):

```
CBEAM_GOLDEN=record pytest tests/integration
git add tests/integration/_golden
```

`CBEAM_GOLDEN=check` is the default; `CBEAM_GOLDEN=off` disables the layer.
A missing `.npz` skips (does not fail) the golden assertion for that test.
