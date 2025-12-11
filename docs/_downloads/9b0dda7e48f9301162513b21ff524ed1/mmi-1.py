from cbeam.waveguide import RectangularStepIndexFiber

mmi_width = 60. # width of the slab's core
mmi_height = 6. # height of the slab's core

access_width = 6. # same for the access waveguide
access_height = 6.

wl = 1.55   # wavelength

ncore = 1.445   # core index
nclad = 1.44    # cladding index

access = RectangularStepIndexFiber(access_width,access_height,access_height*6,access_width*6,ncore,nclad,1.,5.)
mmi = RectangularStepIndexFiber(mmi_width,mmi_height,mmi_width*3,mmi_height*6,ncore,nclad,1.,5.)