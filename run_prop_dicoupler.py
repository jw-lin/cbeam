import numpy as np
import waveguide
import matplotlib.pyplot as plt
from wavesolve.fe_solver import plot_eigenvector,solve_waveguide


def kappa(rcore,ncore,nclad,d,wl):
    k = 2*np.pi/wl
    V = 2*np.pi/wl*rcore*np.sqrt(ncore**2-nclad**2)
    da = d/rcore
    c0 = 5.2789 - 3.663*V + 0.3841 * V**2
    c1 = -0.7769 + 1.2252 * V - 0.0152 * V**2
    c2 = -0.0175 - 0.0064 * V + 0.0009 * V**2
    return np.pi*V/(2*k*nclad*rcore**2) * np.exp(-(c0+c1*da+c2*da**2))

### how to run, a 7-step process ###

# 1. create a Waveguide using optics.py
# 2. initialize a propagator object with the waveguide
# 3. (optional) run get_neffs() to look at effective indices of the eigenmodes and identify degenerate groups
# 4. run prop_setup() to compute everything you need to model the lantern propagation
# 5. (optional) alternatively, load an existing set of computation results with load()
# 6. run make_interp_funcs() to set up the interpolation functions
# 7. run propagate()

## example below ##

# dicoupler params #

wl = 1.55                       # wavelength, um
dmin = 10.
dmax = 60.

rcore = 3.

nclad = 1.444                   # cladding refractive index
ncore = nclad + 8.8e-3          # SM core refractive index

# mesh params #
core_res = 100                      # no. of line segments to use to resolve the core-cladding interface(s)
clad_mesh_size = 40.               # mesh size (triangle side length) to use in the cladding region
core_mesh_size = 1.0               # mesh size (triangle side length) to use inside the cores
size_scale_fac = 0.5
_power = 1.0

# solve params #
tol = 1e-4
degen_groups = [] # these groups remain degenerate throughout our example waveguide
dz0 = 0.5

kap = kappa(rcore,ncore,nclad,dmin,wl)

print(kap)
print(np.pi/kap)

print(np.pi/kappa(rcore,ncore,nclad,10.00030720873011,wl))

neps = 0.

coupling_length = 5000
a = coupling_length/2. # bend length

dicoupler = waveguide.Dicoupler(rcore,rcore+0.5,ncore,ncore+neps,dmax,dmin,nclad,coupling_length,a,core_res,core_mesh_size=core_mesh_size,clad_mesh_size=clad_mesh_size,split=False)

z_ex = dicoupler.z_ex

dicoupler.plot_paths()


tag = "test_dicoupler_asym" # identifier for this computation
import propagator

# 2. initialize the propagator
adprop = propagator.Propagator(wl,dicoupler,2)

# apply meshing params
adprop.mesh_dist_power = _power
adprop.mesh_dist_scale = size_scale_fac
adprop.max_mesh_size = 40


m = adprop.generate_mesh()


"""
_d = dicoupler.assign_IOR()

zts = np.linspace(0,2000,10)

for zt in zts:
    print(zt)
    adprop.wvg.update(zt)
    m = adprop.generate_mesh()
    w,v,n = solve_waveguide(m,wl,_d,sparse=True)
    fig,axs = plt.subplots(3,1)
    dicoupler.plot_mesh(m,_d,ax=axs[0])
    plot_eigenvector(m,v[0],ax=axs[1],show=False)
    plot_eigenvector(m,v[1],ax=axs[2],show=False)
    plt.show()

#m2 = dicoupler.transform_mesh(m,0,2690)

adprop.wvg.update(5000)
m2 = adprop.generate_mesh()
dicoupler.plot_mesh(m2,_d)
"""

#zs,coupling_mats,neffs,vs = adprop.prop_setup(0,z_ex,tol,save=True,tag=tag,degen_groups=degen_groups,min_zstep=2.5,max_zstep=320.,plot=False,mode="transform",dz0=dz0)

adprop.load(tag=tag)

for j in range(2): # plotting cross-coupling matrix
    for i in range(j):
        plt.plot(adprop.za,adprop.cmat[:,i,j],label=str(i)+str(j))
for z in adprop.za:
    plt.axvline(x=z,alpha=0.1,color='k',zorder=-100)
plt.legend(loc='best')
plt.title("cross-coupling matrix")
plt.show()

w = np.power(adprop.neff,2)*adprop.k**2

for i in range(1): # looking at differences in eigenvalue (k^2 neff^2)
    plt.semilogy(adprop.za, adprop.neff[:,i]-adprop.neff[:,i+1],label=str(i)+str(i+1))
    
#plt.semilogy(adprop.za,w[:,0]-w[:,-1],color='k')
plt.axhline(y=1e-4,color='k',ls='dashed')
plt.legend(loc='best')
plt.title("eigenvalue differences")
plt.show()

# 6. make interpolation functions
adprop.make_interp_funcs()

from scipy.integrate import quad

integrand = lambda z: (adprop.neff_funcs[0](z)-adprop.neff_funcs[1](z))

neffint,err = quad(integrand,z_ex/2-coupling_length/2,z_ex/2+coupling_length/2)
print("avg diff neff: ", ((adprop.neff_int_funcs[0](z_ex/2+coupling_length/2) - adprop.neff_int_funcs[0](z_ex/2-coupling_length/2)) - (adprop.neff_int_funcs[1](z_ex/2+coupling_length/2) - adprop.neff_int_funcs[1](z_ex/2-coupling_length/2))) /coupling_length)
print("diff neff at cent: ",adprop.neff_funcs[0](z_ex/2)- adprop.neff_funcs[1](z_ex/2))

u0 = np.array([1,0],dtype=np.complex128) # launch left channel

# 7. propagate
z,u,uf = adprop.propagate(u0,z_ex,WKB=True)

print("final mode amplitudes: ")
print(np.abs(uf))
print("final mode powers: ")
print(np.power(np.abs(uf),2))
print("final total power: ")
print(np.sum(np.power(np.abs(uf),2)))

for i in range(2): # plotting evolution in mode power
    plt.plot(z,np.power(np.abs(u[i]),2),label='mode '+str(i))

plt.plot(z,np.sum(np.power(np.abs(u),2),axis=0),color='k',zorder=-100,label="total power",ls='dashed')
plt.title('LP01 propagation - std PL')

plt.xlabel(r'$z$ (um)')
plt.ylabel("power")

plt.legend(loc='best')
plt.show()

fig,axs = plt.subplots(1,2)
plot_eigenvector(adprop.mesh,adprop.vs[:,0,0],show=False,ax=axs[0])
plot_eigenvector(adprop.mesh,adprop.vs[:,1,0],show=False,ax=axs[1])
plt.show()
_m = adprop.wvg.transform_mesh(adprop.mesh,0,z_ex/2)

fig,axs = plt.subplots(1,2)
plot_eigenvector(_m,adprop.compute_v(z_ex/2)[:,0],show=False,ax=axs[0])
plot_eigenvector(_m,adprop.compute_v(z_ex/2)[:,1],show=False,ax=axs[1])
plt.show()
__m = adprop.wvg.transform_mesh(adprop.mesh,0,z_ex)

fig,axs = plt.subplots(1,2)
plot_eigenvector(__m,adprop.compute_v(z_ex)[:,0],show=False,ax=axs[0])
plot_eigenvector(__m,adprop.compute_v(z_ex)[:,1],show=False,ax=axs[1])
plt.show()


# change the length

z_exs = np.linspace(5000,10000,100)
u0 = np.array([1,0],dtype=np.complex128)
pwrs = []
zcstart = 4999
zcend = 5001
for i,zx in enumerate(z_exs):
    # rescale the region from z=3000 to z= 7000
    za = np.copy(adprop.za)

    za[int(len(za)/2):] += zx
    newza = za

    adprop.make_interp_funcs(newza)

    zs = np.linspace(0,dicoupler.z_ex+zx,1000)
    neffs = adprop.compute_neff(zs)

    z,u,uf = adprop.propagate(u0,newza[-1],WKB=True)
    pwr = np.power(np.abs(uf),2) 
    pwrs.append(pwr)

pwrs = np.array(pwrs)

zmax = z_exs[np.argmax(pwrs[:,0])]
plt.plot(z_exs,np.power(np.cos(kap*(z_exs-zmax)),2),color='k',ls='dashed')


plt.plot(z_exs,pwrs.T[0],label="channel 1")
plt.plot(z_exs,pwrs.T[1],label="channel 2")
plt.legend(loc='best',frameon=False)
plt.xlabel("dicoupler length")
plt.ylabel("power")
plt.show()
