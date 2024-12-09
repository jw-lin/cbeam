betas = mmi_prop.neffs[0] * 2 * np.pi / wl
L = np.pi/(betas[0]-betas[1])

f = mmi_prop.make_field(launch_modes,z=L/4,apply_phase=True)

mmi_prop.plot_cfield(f,xlim=(-30,30),ylim=(-30,30),show_mesh=True)

# you could also do
# zs,us,uf = mmi_prop.propagate(launch_mvec,0,L/4)
# f = prop.make_field(uf,apply_phase=False)