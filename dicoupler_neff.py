import numpy as np
import waveguide
import matplotlib.pyplot as plt
from wavesolve.fe_solver import plot_eigenvector,solve_waveguide,get_eff_index


def kappa(rcore,ncore,nclad,d,wl):
    k = 2*np.pi/wl
    V = 2*np.pi/wl*rcore*np.sqrt(ncore**2-nclad**2)
    da = d/rcore
    c0 = 5.2789 - 3.663*V + 0.3841 * V**2
    c1 = -0.7769 + 1.2252 * V - 0.0152 * V**2
    c2 = -0.0175 - 0.0064 * V - 0.0009 * V**2
    return np.pi*V/(2*k*nclad*rcore**2) * np.exp(-(c0+c1*da+c2*da**2))

wl = 1.55                       # wavelength, um
dmin = 10.
dmax = 10.

rcore = 3.
rclad = 40

nclad = 1.444                   # cladding refractive index
ncore = nclad + 8.8e-3          # SM core refractive index

# mesh params #
core_res = 60                      # no. of line segments to use to resolve the core-cladding interface(s)
clad_res = 20
clad_mesh_size = 10.               # mesh size (triangle side length) to use in the cladding region
core_mesh_size = 0.25               # mesh size (triangle side length) to use inside the cores
size_scale_fac = 1.0
_power = 1.0

# solve params #
tol = 1e-4
degen_groups = [] # these groups remain degenerate throughout our example waveguide
dz0 = 0.5

neps = 0

coupling_length = 5000
a = coupling_length/4. # bend length

#dicoupler = waveguide.Dicoupler(rcore,rcore,ncore,ncore+neps,dmax,dmin,nclad,coupling_length,a,core_res,core_mesh_size=core_mesh_size,clad_mesh_size=clad_mesh_size,split=False)

dicoupler = waveguide.CircDicoupler(rcore,rcore,ncore,ncore+neps,dmax,dmin,nclad,coupling_length,a,core_res,core_mesh_size=core_mesh_size,clad_mesh_size=clad_mesh_size,rclad=rclad,clad_res=clad_res)

z_ex = dicoupler.z_ex

import propagator

# 2. initialize the propagator
adprop = propagator.Propagator(wl,dicoupler,2)

# apply meshing params
adprop.mesh_dist_power = _power
adprop.mesh_dist_scale = size_scale_fac
adprop.max_mesh_size = 40


m = adprop.generate_mesh()
print(m.points.shape)

dicoupler.plot_mesh(m)

adprop.wvg.update(0)
_m = m#dicoupler.transform_mesh(m,0,5000)


w,v,n = solve_waveguide(_m,wl,dicoupler.assign_IOR(),sparse=True)
print(get_eff_index(wl,w[0]))
print( get_eff_index(wl,w[0])-get_eff_index(wl,w[1]))

fig,axs = plt.subplots(1,2)
plot_eigenvector(_m,v[0],ax=axs[0],show=False)
plot_eigenvector(_m,v[1],ax=axs[1],show=False)
plt.show()