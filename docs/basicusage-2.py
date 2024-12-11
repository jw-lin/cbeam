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
prop.characterize(save=True,tag=tag)

# just use the the following if you already ran the above
# prop.load(tag)

# propagation

u0 = [1,0,0,0,0,0] # starting mode vector, corresponding to fundamental mode

zs,us,uf = prop.propagate(u0) # default behavior is to propagate through the entire waveguide

# get the fields and plot

input_field = prop.make_field(u0,0)
output_field = prop.make_field(us[:,-1],zs[-1])

import matplotlib.pyplot as plt
fig,axs = plt.subplots(1,2,)

prop.plot_cfield(input_field,z=0,ax=axs[0],show_mesh=True)
prop.plot_cfield(output_field,z=zs[-1],ax=axs[1],show_mesh=True,xlim=(-100,100),ylim=(-100,100))

# plot decoration
axs[0].set_title("initial field")
axs[1].set_title("final field")

# must manually call show() if you're plotting on a premade axis
plt.show()