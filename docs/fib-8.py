# solve for top 4 modes in terms of effective index
rect_prop = Propagator(wavelength,rect_fiber,4)

rect_tag = "tapered_box"

# uncomment below as necessary
# rect_prop.compute_modes(0,length,save=True,tag=rect_tag)
rect_prop.load(rect_tag)