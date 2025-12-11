rect_clad.mesh_size = 3. # target size of triangles in cladding (but not core)
rect_core.mesh_size = 1. # target size in core

rect_fiber = waveguide.Waveguide([rect_clad,rect_core])