# Overview

this is (will be) a propagator for slowly-varying and weakly-guiding waveguides (quasi-adiabatic, hence the "slow"). kind of ironic because if everything goes to plan this should be way faster than `lightbeam` lol.

## details

this package uses a coupled-mode approach to simulate propagation through waveguides. currently, only tapered waveguides, whose cross-sectional geometries are assumed to scale uniformly with propagation distance, are supported. eigenmodes are computed using `wavesolve`, a finite element mode solver. meshes are generated using `gmsh` and `pygmsh`, and support boundary layer refinement at interfaces between materials with different refractive indices. derivatives of eigenmodes are estimated using centered finite difference. to interpolate quickly between different finite element meshes, `slowbeam` includes a small `Julia` package which accelerates mesh point queries by storing triangles in a bounding volume hierarchy (BVH) tree. 
