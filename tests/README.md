# cbeam test-suite

```
tests/
├── conftest.py          shared fixtures + one-time FEval (Julia) project setup
├── unit/                fast, isolated tests of individual classes / functions
│   ├── test_misc.py                helper functions in cbeam.waveguide
│   ├── test_prim2d.py              Prim2D, Circle, Rectangle, Prim2DUnion
│   ├── test_prim3d.py              Prim3D, Pipe, LinearPipe, Box, BoxPipe
│   ├── test_waveguide.py           the base Waveguide class (mesh, transform, IOR)
│   ├── test_waveguide_classes.py   every pre-defined Waveguide subclass
│   ├── test_feval.py               cbeam.FEval (Julia-backed field evaluator)
│   └── test_propagator.py          Propagator / ChainPropagator helpers + z-invariant path
└── integration/         end-to-end tests, each a rewrite of a documentation example
    ├── test_quickstart.py          docs_source/basicusage.rst
    ├── test_fib.py                 docs_source/fib.rst
    ├── test_photonic_lantern.py    docs_source/PL.rst
    ├── test_dicoupler.py           docs_source/dc.rst
    ├── test_mmi.py                 docs_source/mmi.rst
    └── test_pl19.py                docs_source/PL19.rst
```

## Running

```bash
pytest -m "not slow"     # unit + light integration, ~30 s
pytest                    # everything, several minutes
pytest -m slow            # only the full characterizations / propagations
```

`slow` marks any test that runs a full adaptive `Propagator.characterize` /
`compute_modes` / `compute_neffs` on a tapering waveguide.  The slow integration
tests deliberately shorten the devices from the documentation (e.g. a 4 mm slice
of the 40 mm photonic lantern) — the numerical code path is identical, only the
number of z-steps changes — and assert on physical invariants (power
conservation, mode ordering, matrix (anti)symmetry) rather than the exact
numbers printed in the docs, which depend on the Gmsh mesh.

## Requirements

The suite needs the full runtime stack: `numpy`, `scipy`, `matplotlib`,
`pygmsh` + `gmsh`, `meshio`, `wavesolve`, and `juliacall` with a working Julia.
The first run instantiates the small Julia project in `src/cbeam/FEval`
(generates `Manifest.toml`); this is done automatically in `conftest.py`.

Tests import `cbeam` from `./src` (configured via `pythonpath` in
`pyproject.toml`), so they always exercise the code in this checkout regardless
of any installed copy.
