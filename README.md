# overview

this is (will be) a propagator for slowly-varying and weakly-guiding waveguides.

## dependencies
Python: `numpy`,`scipy`,`juliacall`,`wavesolve`,`pygmsh`,`meshio`,`matplotlib`

Julia: `pythoncall`

General: `Gmsh`

## details

this package uses a coupled-mode approach to simulate propagation through waveguides (hence the name). some notes: 

1. currently, only tapered waveguides, whose cross-sectional geometries are assumed to scale uniformly with propagation distance, are supported.
2. eigenmodes are computed using `wavesolve`, a finite element mode solver, which in turn uses a sparse method from `scipy` to solve the generalized eigenvalue problem.
3. meshes are generated using `Gmsh` and `pygmsh`; boundary layer refinement at interfaces between regions with different refractive index is supported.
4. derivatives of eigenmodes are estimated using centered finite difference (as opposed to perturbation theory).
5. the $z$ step size is chosen adaptively, by comparing values at a proposed next $z$ step with an prediction obtained by extrapolating from previous values.
6. to interpolate quickly between different finite element meshes, `coupledbeam` includes a small Julia package called `FEinterp` which accelerates mesh point queries by storing mesh triangles in a bounding volume hierarchy (BVH) tree. 

(aside: interpolation really isn't the right word for #6, since a field defined over a finite element mesh has a definite value at every point within the mesh, not just at the mesh nodes. you just have to find which mesh triangle contains a given point).

## installation help
the `pythoncall` installation is a little weird. once you have the code cloned (and assuming you have a Julia install), go into `coupledbeam`'s root directory and start up a Julia REPL. from here run

```
using Pkg
Pkg.activate("FEinterp")
Pkg.add("PythonCall")
exit()
```

and then you should be good to go.
