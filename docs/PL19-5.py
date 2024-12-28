# from the previous analysis, we only need to track the top 20 modes
# to ensure that all guided modes are tracked
prop1 = Propagator(wl,PL19,20)

prop1.degen_groups = [[1,2],[3,4],[6,7],[8,9],[10,11],[12,13],[15,16]]

# during characterization, we specify modes we don't care about
# to speed things up. mode 18 becomes a cladding mode, as
# per previous analysis.
prop1.skipped_modes = [18]

tag1 = "19port_0800_front"
#prop1.characterize(0,50000,save=True,tag=tag1)
prop1.load(tag1)

# prop1.compute_neffs(tag=tag1,save=True)
prop1.load(tag=tag1)
prop1.plot_coupling_coeffs(legend=False)