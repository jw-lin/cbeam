from cbeam.propagator import Propagator

access_prop = Propagator(wl,access,Nmax=1)
ac_neffs,ac_modes = access_prop.solve_at(0)

access_prop.plot_cfield(ac_modes[0],show_mesh=True)