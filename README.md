# Overview

`cbeam` is a propagator for slowly varying and weakly guiding waveguides, written in Python and Julia. 

Now with website [documentation](https://jw-lin.github.io/cbeam/)!

Current version: v0.0.4

## Dependencies
Python: `numpy`,`scipy`,`juliacall`,`wavesolve`,`pygmsh`,`meshio`,`matplotlib`

Julia: `pythoncall`

General: `Gmsh`

## Details

This package uses coupled-mode theory to simulate propagation through waveguides, under the weak guidance, paraxial, and slowly varying envelope approximations. An end-to-end characterization of a simple waveguide (2x2 directional coupler, few-mode photonic lantern) takes a few minutes. Wavefronts are decomposed in the basis of instantaneous eigenmodes, so that `cbeam` remains applicable even when a static eigenbasis does not exist (the case for tapered waveguides such the photonic lantern). It's also fast; a 6-port lantern can be characterized in under a minute on (my) laptop hardware, and a directional coupler takes around 10 s. See the online documentation for installation instructions, examples, and further reference. For a brief writeup of the theory behind `cbeam`, check out this [arXiv paper](https://arxiv.org/abs/2411.08118).

## Acknowledgments
NSF grants 2109231, 2109232, 2308360, 2308361
