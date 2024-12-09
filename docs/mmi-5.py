from cbeam import FEval

# resample takes: input field, input mesh, output mesh
launch_field = FEval.resample(ac_modes[0],access_prop.mesh,mmi_prop.mesh)

# we'll plot to make sure it looks good
mmi_prop.plot_cfield(launch_field,xlim=(-16,16),ylim=(-16,16),show_mesh=True)