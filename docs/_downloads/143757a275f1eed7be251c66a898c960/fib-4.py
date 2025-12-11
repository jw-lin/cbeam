from cbeam.propagator import Propagator
wavelength = 1.55 # um
Nmax = 10 # solve for 10 modes
prop = Propagator(wavelength,fiber,Nmax)
effective_indices, modes = prop.solve_at(0) # could solve at any z value too
prop.plot_cfield(modes[1],res=0.5) # look at LP_11