wl = 0.8                         # wavelength, um
taper_factor = 12.               # relative scale factor between frontside and backside waveguide geometry
rcore = 1.8/taper_factor         # radius of tapered-down single-mode cores at frontside (small end), um
rclad = 9.0                      # radius of cladding-jacket interface at frontside, um
rjack = 27                       # radius of outer jacket boundary at frontside, um
z_ex = 100000                    # lantern length, um

nclad = 1.444                    # cladding refractive index
ncore = nclad + 8.8e-3           # SM core refractive index
njack = nclad - 5.5e-3           # jacket (low-index capillary) refractive index

rcores = [rcore]*19
ncores = [ncore]*19