# overview

`cbeam` is a propagator for slowly-varying and weakly-guiding waveguides, written in Python and Julia.

## dependencies
Python: `numpy`,`scipy`,`juliacall`,`wavesolve`,`pygmsh`,`meshio`,`matplotlib`

Julia: `pythoncall`,`StaticArrays`,`GrundmannMoeller`

General: `Gmsh`

## details

this package uses coupled-mode thoery to simulate propagation through waveguides, under the weak guidance, paraxial, and slowly-varying envelope approximations. See `cbeam.ipynb` for installation instructions, documentation, and worked examples. For a brief writeup of the theory behind `cbeam`, check out `coupled_mode_theory.pdf`.
