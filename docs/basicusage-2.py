from cbeam.propagator import Propagator
from cbeam.waveguide import TestPhotonicLantern

# make the waveguide
wvg = TestPhotonicLantern()

wavelength = 1.55 # um
num_modes = 6 # assuming we're using the 6-port lantern from earlier
tag = "test"

# make the propagator
prop = Propagator(wavelength,wvg,num_modes)

# characterization - comment/uncomment below as needed
# prop.characterize(save=True,tag=tag)
prop.load(tag=tag)

u0 = [1,0,0,0,0,0] # starting mode vector, corresponding to fundamental mode

# this returns the z values, coupled-mode amplitudes, and the output amplitudes
zs,us,uf = prop.propagate(u0)