# overview

this is (will be) a propagator for slowly-varying and weakly-guiding waveguides.

## dependencies
Python: `numpy`,`scipy`,`juliacall`,`wavesolve`,`pygmsh`,`meshio`,`matplotlib`

Julia: `pythoncall`

General: `Gmsh`

## details

this package uses a coupled-mode approach to simulate propagation through waveguides (hence the name). some notes: 

1. eigenmodes are computed using `wavesolve`, a finite element mode solver, which in turn uses a sparse method from `scipy` to solve the generalized eigenvalue problem.
2. meshes are generated using `Gmsh` and `pygmsh`; boundary layer refinement at interfaces between regions with different refractive index is supported.
3. meshes must evolve continuously with $z$, following the $z$-variation of the waveguide. this evolution is waveguide-specific and as of now user-defined (which makes it difficult to simulate some kinds of waveguides, at least until i come up with a more general solution.)
4. derivatives of eigenmodes are estimated using centered finite difference (as opposed to perturbation theory).
5. the $z$ step size is chosen adaptively, by comparing values at a proposed next $z$ step with an extrapolation from previous values.
6. to quickly evaluate a field sampled on a finite element mesh at an arbitrary $(x,y)$ point, `coupledbeam` includes a small Julia package called `FEval`, which speeds up point queries by storing mesh triangles in a bounding volume hierarchy (BVH) tree.

See `cbeam.ipynb` for installation instructions, documentation, and worked examples. 
