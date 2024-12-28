tag2 = "19port_0800_back"
prop2 = Propagator(wl,PL19,20)
prop2.skipped_modes = [18]

# every mode except 18 is degenerate w/ each other
prop2.degen_groups = [[i for i in range(20)]]
del prop2.degen_groups[0][18]

# use the final modes of prop1 as the
# initial modes of prop2
prop2.load_init_conds(prop1)

# prop2.characterize(50000,100000,save=True,tag=tag2)
prop2.load(tag2)

prop2.plot_coupling_coeffs(legend=False)