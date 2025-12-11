from cbeam import waveguide

# make a waveguide for testing - 6 port photonic lantern
wvg = waveguide.TestPhotonicLantern()
mesh = wvg.make_mesh()

wvg.plot_mesh(mesh=mesh) # can also leave out mesh; a mesh will be auto-generated