# solve for top 6 modes in terms of effective index
rect_prop = Propagator(wavelength,rect_fiber,6)

rect_tag = "tapered_box"

# comment/uncomment below as necessary
rect_prop.compute_neffs(0,length,save=True,tag=rect_tag)
# rect_prop.load(rect_tag)

# if you wanted a more careful computation of the modes, you could also use
# rect_prop.compute_modes()