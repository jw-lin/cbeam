from cbeam.propagator import Propagator
wavelength = 1.55 # um
PLprop = Propagator(wavelength,lant,6) # 6 modes

PLneff,PLmodes = PLprop.solve_at(z=z_ex/2.)
PLprop.plot_cfield(PLmodes[0],z=z_ex/2.)