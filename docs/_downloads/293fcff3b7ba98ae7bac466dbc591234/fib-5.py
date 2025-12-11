length = 10000 # um
xw_core = lambda z: 10 * (1+2*z/length) # triples in thickness over length
yw_core = 10 # fixed height

rect_core = waveguide.BoxPipe(ncore,"core",xw_core,yw_core)

xw_clad = lambda z: 30 * (1+2*z/length)

# cladding will just be a bigger core
rect_clad = waveguide.BoxPipe(nclad,"clad",xw_clad,3*yw_core)