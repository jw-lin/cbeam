# Overview

this is (will be) a propagator for slowly-varying and weakly-guiding waveguides (quasi-adiabatic, hence the "slow"). kind of ironic because if everything goes to plan this should be way faster than `lightbeam` lol.

## dependencies
Python: `numpy`,`scipy`,`juliacall`,`wavesolve`,`pygmsh`,`meshio`,`matplotlib`

Julia: `pythoncall`

General: `Gmsh`

## details

this package uses a coupled-mode approach to simulate propagation through waveguides. some notes: 

1. currently, only tapered waveguides, whose cross-sectional geometries are assumed to scale uniformly with propagation distance, are supported.
2. eigenmodes are computed using `wavesolve`, a finite element mode solver, which in turn uses a sparse solver from `scipy` to solve the generalized eigenvalue problem.
3. meshes are generated using `Gmsh` and `pygmsh`; boundary layer refinement at interfaces between regions with different refractive index is supported.
4. derivatives of eigenmodes are estimated using centered finite difference (as opposed to perturbation theory).
5. to interpolate quickly between different finite element meshes, `slowbeam` includes a small `Julia` package which accelerates mesh point queries by storing mesh triangles in a bounding volume hierarchy (BVH) tree. 
