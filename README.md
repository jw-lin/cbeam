# Overview

`cbeam` is a propagator for slowly-varying and weakly-guiding waveguides, written in Python and Julia.

## Dependencies
Python: `numpy`,`scipy`,`juliacall`,`wavesolve`,`pygmsh`,`meshio`,`matplotlib`

Julia: `pythoncall`,`StaticArrays`,`GrundmannMoeller`,`Cubature`

General: `Gmsh`

## Details

This package uses coupled-mode theory to simulate propagation through waveguides, under the weak guidance, paraxial, and slowly-varying envelope approximations. Wavefronts are decomposed in the basis of instantaneous eigenmodes, so that `cbeam` remains applicable even when a static eigenbasis does not exist (the case for tapered waveguides such the photonic lantern). See `cbeam.ipynb` for installation instructions, documentation, and worked examples. For a brief writeup of the theory behind `cbeam`, check out `coupled_mode_theory.pdf`.
