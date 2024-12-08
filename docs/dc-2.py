from cbeam import waveguide

dicoupler = waveguide.Dicoupler(rcore,rcore,ncore,ncore,dmax,dmin,nclad,coupling_length,bend_length,core_res,core_mesh_size=core_mesh_size,clad_mesh_size=clad_mesh_size)

# here we could adjust the boundary refinement mesh parameters, e.g.
# dicoupler.min_mesh_size = <minimum mesh element size>

dicoupler.plot_paths()