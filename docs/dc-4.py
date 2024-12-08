from cbeam import propagator
dc_prop = propagator.Propagator(wl,dicoupler,Nmax=2)

neff,modes = dc_prop.solve_at(z=dicoupler.z_ex/2)
dc_prop.plot_cfield(modes[1],z=dicoupler.z_ex/2)