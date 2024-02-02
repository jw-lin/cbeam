### lantern propagation sim params ###
import matplotlib.pyplot as plt
import numpy as np

wl = 1.55                       # wavelength, um
taper_factor = 8.               # relative scale factor between frontside and backside waveguide geometry    
rclad = 10                      # radius of cladding-jacket interface at frontside, um
rjack = 30                      # radius of outer jacket boundary at frontside, um
z_ex = 100000                    # lantern length, um

ring_radius = 10
ring_width = 8

nclad = 1.444 + 0.0295                 # cladding refractive index
ncore = 1.444 + 0.0335          # SM core refractive index
njack = 1.444 - 0.0009          # jacket (low-index capillary) refractive index

# mesh params #

core_res = 30                      # no. of line segments to use to resolve the core-cladding interface(s)
clad_res = 60                      # no. of line segments to use to resolve the cladding-jacket interface
jack_res = 30                      # no. of line segments to form the outer jacket boundary
clad_mesh_size = 5.0               # mesh size (triangle side length) to use in the cladding region
core_mesh_size = 0.2               # mesh size (triangle side length) to use inside the cores
inner_clad_mesh_size = 10.

rcores = np.array([13/2,8.8/2,10.2/2,11.6/2,11.6/2,10.2/2,8.8/2])/taper_factor              # core radii
ncores = [ncore]*7                  # core index

tag = "OAMPL"               # identifier for this model

import waveguide,propagator
from wavesolve.fe_solver import solve_waveguide

OAMPL = waveguide.OAMPhotonicLantern(ring_radius,ring_width,rcores,rjack,ncores,nclad,njack,z_ex,taper_factor,core_res,clad_res,clad_res,jack_res,core_mesh_size,clad_mesh_size,inner_clad_mesh_size)

#m = OAMPL.make_mesh_bndry_ref()
#m2 = OAMPL.transform_mesh(m,0,z_ex)
#_d = OAMPL.assign_IOR()

#solve_waveguide(m2,wl,_d,plot=True,sparse=True)


OAMPL.mesh_dist_scale = 1.0

prop = propagator.Propagator(wl,OAMPL,7)
prop.max_zstep = 320*4
prop.zstep_tol = 2.5e-5
prop.min_zstep = 10/16

prop.prop_setup(0,z_ex,save=True,tag=tag,plot=True)

prop.load(tag)

prop.make_interp_funcs()

for i in range(7):
    plt.plot(prop.za,prop.neff[:,i])
plt.show()

for i in range(1,7):
    plt.semilogy(prop.za,prop.neff[:,0]-prop.neff[:,i])
plt.show()
from wavesolve.fe_solver import plot_eigenvector
#for i in range(5):
#    plot_eigenvector(prop.mesh,prop.vs[0,i,:])

prop.plot_coupling_coeffs()
u0 = np.array([0,0,0,0,0,1,1j])/np.sqrt(2)
z,u,uf = prop.backpropagate(u0,z_ex,0)

for i in range(7):
    plt.plot(z,np.abs(u[i]),label="mode "+str(i))
plt.legend(loc='best')
plt.show()

f = prop.make_field(uf,0)

#plt.tricontourf(prop.mesh.points[:,0],prop.mesh.points[:,1],np.angle(f2),cmap='hsv',vmin=-np.pi,vmax=np.pi,levels=60)
#plt.show()
#prop.plot_field(np.abs(f),0)
prop.plot_cfield(f,0,show_mesh=True)