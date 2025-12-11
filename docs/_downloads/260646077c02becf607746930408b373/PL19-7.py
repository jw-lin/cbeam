from cbeam.propagator import ChainPropagator
prop12 = ChainPropagator([prop1,prop2])

u0 = [0.]*20
# launch LP01
u0[0] = 1.

# propagate as normal
zs,us,uf = prop12.propagate(u0)

prop12.plot_mode_powers(zs,us)