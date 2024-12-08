from cbeam import waveguide
rcore,rclad = 10,30 # units are in whatever we choose for wavelength later, which will be um.
ncore,nclad = 1.445,1.44
res = 30 # resolution
core = waveguide.Pipe(ncore,"core",res,rcore)
clad = waveguide.Pipe(nclad,"clad",3*res,rclad)