from cbeam.waveguide import PhotonicLantern
import numpy as np

wl = 1.55                       # wavelength, um
taper_factor = 8.               # relative scale factor between frontside and backside waveguide geometry
rcore = 2.2/taper_factor        # radius of tapered-down single-mode cores at frontside (small end), um
rclad = 10                      # radius of cladding-jacket interface at frontside, um
rjack = 30                      # radius of outer jacket boundary at frontside, um
z_ex = 40000                    # lantern length, um

nclad = 1.444                   # cladding refractive index
ncore = nclad + 8.8e-3          # SM core refractive index
njack = nclad - 5.5e-3          # jacket (low-index capillary) refractive index

t = 2*np.pi/5
core_offset = rclad*2/3         # offset of outer ring of cores from center

# initial core positions. the first core is at (0,0) and the other 5 form a pentagon at a distance <core_offset> from the center
core_pos = np.array([[0,0]] + [[core_offset*np.cos(i*t),core_offset*np.sin(i*t)] for i in range(5)])

# we'll leave the optional args at their defaults
# below is actually equivalent to TestPhotonicLantern
lant = PhotonicLantern(core_pos,[rcore]*6,rclad,rjack,[ncore]*6,nclad,njack,z_ex,taper_factor)