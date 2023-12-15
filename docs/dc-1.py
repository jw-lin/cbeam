### symmetric dicoupler propagation parameters ###

wl = 1.55                           # wavelength, um
dmin = 10.                          # minimum center-to-center separation of single mode channels
dmax = 60.                          # maximum center-to-center separation of single mode channels
coupling_length = 5000.             # length of coupling region (where SM channels are close)
bend_length = coupling_length/4.   # approximate length of channel bends

rcore = 3.                      # core radius. we will simulate a symmetric dicoupler, so core radii of both channels are the same

nclad = 1.444                   # cladding refractive index
ncore = nclad + 8.8e-3          # SM core refractive index

# mesh params #
core_res = 20                    # no. of line segments to use to resolve the core-cladding interface(s)
clad_mesh_size = 20.0               # mesh size (triangle side length) to use in the cladding region
core_mesh_size = 1.0                # mesh size (triangle side length) to use inside the cores

tag = "test_dicoupler"