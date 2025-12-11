# get the fields and plot
output_field = prop.make_field(us[-1],zs[-1])

prop.plot_cfield(output_field,z=zs[-1],show_mesh=True,xlim=(-100,100),ylim=(-100,100),res=0.5)