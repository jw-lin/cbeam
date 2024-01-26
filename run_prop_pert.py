import numpy as np
import waveguide
import matplotlib.pyplot as plt
from wavesolve.fe_solver import plot_eigenvector

### how to run, a 7-step process ###

# 1. create a Waveguide using optics.py
# 2. initialize a propagator object with the waveguide
# 3. (optional) run get_neffs() to look at effective indices of the eigenmodes and identify degenerate groups
# 4. run prop_setup() to compute everything you need to model the lantern propagation
# 5. (optional) alternatively, load an existing set of computation results with load()
# 6. run make_interp_funcs() to set up the interpolation functions
# 7. run propagate()

## example below ##

# lantern params #

wl = 1.55                       # wavelength, um
taper_factor = 8.                # relative scale factor between frontside and backside waveguide geometry    
rcore = 2.2/taper_factor        # radius of tapered-down single-mode cores at frontside, um
rclad = 10                      # radius of cladding-jacket interface at frontside, um
rjack = 40                      # radius of outer jacket boundary at frontside, um
z_ex = 20000                    # lantern length, um

nclad = 1.444                   # cladding refractive index
ncore = nclad + 8.8e-3          # SM core refractive index
njack = nclad - 5.5e-3          # jacket refractive index

t = 2*np.pi/5                   
initial_offset = rclad*2/3
xpos_i = [0,initial_offset,initial_offset*np.cos(t),initial_offset*np.cos(2*t),initial_offset*np.cos(3*t),initial_offset*np.cos(4*t)]
ypos_i = [0,0,initial_offset*np.sin(t),initial_offset*np.sin(2*t),initial_offset*np.sin(3*t),initial_offset*np.sin(4*t)]

core_pos = np.array([xpos_i,ypos_i]).T  # core positions for a standard 6 port PL at frontside
# mesh params #

core_res = 50                      # no. of line segments to use to resolve the core-cladding interface(s)
clad_res = 150                      # no. of line segments to use to resolve the cladding-jacket interface
jack_res = 30                       # no. of line segments to form the outer jacket boundary
clad_mesh_size = 5.0               # mesh size (triangle side length) to use in the cladding region
core_mesh_size = 0.05               # mesh size (triangle side length) to use inside the cores

max_zstep = 320
min_zstep = 1.25/4

# solve params #
tol = 5e-5
degen_groups = []#[[1,2],[3,4]] # these groups remain degenerate throughout our example waveguide
dz0 = 0.1

std_rcores = np.array([rcore]*6)

neps = 0#1e-6
reps = 1e-4
ncores = [ncore + i*neps for i in range(6)]
rcores = [rcore + i*reps for i in range(6)]
# 1. create a Waveguide (standard lantern)
lant = waveguide.PhotonicLantern(core_pos,rcores,rclad,rjack,ncores,nclad,njack,z_ex,taper_factor,core_res,clad_res,jack_res,core_mesh_size,clad_mesh_size)

tag = "0_std_test_pert" # identifier for this computation
import propagator

# 2. initialize the propagator
adprop = propagator.Propagator(wl,lant,6)

adprop.mesh_dist_power = 1.
adprop.mesh_dist_scale = 1.0
adprop.max_mesh_size = 40
adprop.min_mesh_size = 0.05

# 4. run prop_setup()
#zs,coupling_mats,neffs,vs = adprop.prop_setup(0,z_ex,tol,save=True,tag=tag,degen_groups=degen_groups,min_zstep=min_zstep,max_zstep=max_zstep,plot=True,dz0=dz0)

ps = "_"+tag if tag is not None else ""

meshwriteto="./data/meshes/mesh"+ps

mesh0,IORdict = lant.make_intersection_mesh(0,dz0,adprop.mesh_dist_scale,adprop.mesh_dist_power,adprop.min_mesh_size,adprop.max_mesh_size,writeto=meshwriteto)
IORsq_diff_dict = lant.IORsq_diff(IORdict)
print(IORdict)
zs,coupling_mats,neffs,vs = adprop.prop_setup_pert(0,z_ex,mesh0,IORdict,IORsq_diff_dict,tol,dz0,min_zstep,True,tag,max_zstep,True,min_degen_dif=0)

# 5. load the results of prop_setup() from local
adprop.load(tag=tag)

#adprop.mesh = adprop.generate_mesh()

print("initial modes: ")
fig,axs = plt.subplots(2,3,sharex=True,sharey=True)
plot_eigenvector(adprop.mesh,adprop.vs[:,0,0],ax=axs[0,0],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,1,0],ax=axs[0,1],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,2,0],ax=axs[0,2],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,3,0],ax=axs[1,0],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,4,0],ax=axs[1,1],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,5,0],ax=axs[1,2],show=False)
plt.show()


print("launch: ")
ui = np.array([0,0,0,0,0,1])
vi = np.sum(adprop.vs[:,:,0]*ui,axis=1)

plot_eigenvector(adprop.mesh,vi)

for j in range(6): # plotting cross-coupling matrix
    for i in range(j):
        plt.plot(adprop.za,adprop.cmat[:,i,j],label=str(i)+str(j))
for z in adprop.za:
    plt.axvline(x=z,alpha=0.1,color='k',zorder=-100)
plt.legend(loc='best')
plt.title("cross-coupling matrix")
plt.show()

w = np.power(adprop.neff,2)*adprop.k**2

for i in range(5): # looking at differences in eigenvalue (k^2 neff^2)
    plt.semilogy(adprop.za, w[:,i]-w[:,i+1],label=str(i)+str(i+1))
    
plt.semilogy(adprop.za,w[:,0]-w[:,-1],color='k')
plt.axhline(y=1e-4,color='k',ls='dashed')
plt.legend(loc='best')
plt.title("eigenvalue differences")
plt.show()

# 6. make interpolation functions
adprop.make_interp_funcs()

u0 = np.array(ui,dtype=np.complex128) # launch 

# 7. propagate
z,u,uf = adprop.propagate(u0,z_ex,WKB=False)

print("final mode amplitudes: ")
print(np.abs(uf))
print("final mode powers: ")
print(np.power(np.abs(uf),2))
print("final total power: ")
print(np.sum(np.power(np.abs(uf),2)))

for i in range(6): # plotting evolution in mode power
    plt.plot(z,np.power(np.abs(u[i]),2),label='mode '+str(i))

plt.plot(z,np.sum(np.power(np.abs(u),2),axis=0),color='k',zorder=-100,label="total power",ls='dashed')
plt.title('LP01 propagation - std PL')

plt.xlabel(r'$z$ (um)')
plt.ylabel("power")

plt.legend(loc='best')
plt.show()

fig,axs = plt.subplots(1,2) # plotting evolution in real and imaginary part of mode amplitude
for i in range(6):
    axs[0].plot(z,np.real(u[i]),label='mode '+str(i))
    axs[1].plot(z,np.imag(u[i]),label='mode '+str(i))
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
plt.show()

print("final modes: ")
fig,axs = plt.subplots(2,3,sharex=True,sharey=True)
plot_eigenvector(adprop.mesh,adprop.vs[:,0,-1],ax=axs[0,0],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,1,-1],ax=axs[0,1],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,2,-1],ax=axs[0,2],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,3,-1],ax=axs[1,0],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,4,-1],ax=axs[1,1],show=False)
plot_eigenvector(adprop.mesh,adprop.vs[:,5,-1],ax=axs[1,2],show=False)
plt.show()


print("final field: ")
fig,axs = plt.subplots(1,3,sharey=True)
_v = np.sum(adprop.vs[:,:,-1]*uf[None,:],axis=1)
plot_eigenvector(adprop.mesh,np.real(_v),ax=axs[0],show=False)
plot_eigenvector(adprop.mesh,np.imag(_v),ax=axs[1],show=False)
plot_eigenvector(adprop.mesh,np.abs(_v),ax=axs[2],show=False)
axs[0].set_title("real part")
axs[1].set_title("imag part")
axs[2].set_title("norm")
plt.show()

# change of basis
from wavesolve.fe_solver import solve_waveguide
from wavesolve.mesher import plot_mesh

_v = np.zeros((adprop.Nmax,adprop.vs.shape[0]))
adprop.wvg.assign_IOR()
m = adprop.wvg.transform_mesh(adprop.mesh,0,z_ex)

for i in range(6):
    _dict = adprop.wvg.isolate_isect(i,IORdict)
    _wi,_vi,_Ni = solve_waveguide(m,wl,_dict,sparse=True,Nmax=1)
    _v[i,:] = _vi


cob,_uf = adprop.compute_change_of_basis(_v.T,z_ex,m,u=uf)

_us = []
for i,_z in enumerate(z):
    _us.append(np.dot(cob,u[:,i]))
_us = np.array(_us)

final_phase = np.exp(1.j*(adprop.k*np.array(adprop.compute_int_neff(z_ex)-adprop.compute_int_neff(0))))
fig,axs = plt.subplots(1,3) # plotting evolution in real and imaginary part of mode amplitude
for i in range(6):
    axs[0].plot(z,np.real(_us[:,i]*final_phase[None,i] ),label='mode '+str(i))
    axs[1].plot(z,np.imag(_us[:,i]*final_phase[None,i] ),label='mode '+str(i))
    axs[2].plot(z,np.abs(_us[:,i]),label='mode '+str(i))
    axs[0].legend(loc='best')
plt.show()

print("channel powers: ")
print(np.power(np.abs(_uf),2))

### note keeping please ignore

## MSPL
# ms_rcores = np.array([7.35/2,8.5/2,8.5/2,9.6/2,9.6/2,10.7/2])/taper_factor

## standard PL result notes ... 
# tag '0' core res 50 clad res 150 jack res 30 clad mesh size 1 interp error 5e-4 : 1588 seconds
# tag '1' core res 40 -> output is not fully symmetrcic : 824 seconds
# tag '2' core res 50 clad res 120 -> output is not fully symmetric : 796 seconds
# tag '3' same as std0 i think ? : 1450 seconds
# tag '4' interp error 1e-3 -> output is not fully symmetric (worst so far): 1223 seconds
# tag '5' core res 60 : 1139 seconds -> max ~0.01 difference in power between outer ports
# tag '6' core res 60 interp error 2.5e-4 : 1512 seconds -> max ~0.005 difference (1244 seconds on desktop)
# tag '7' core res 60 interp error 1e-4 : 1467 seconds -> max ~0.0037 difference
# tag '8' core res 60 clad red 180 interp error 1e-4 -> max ~0.0021 difference (1470 seconds)
# tag '9' core res 80 clad red 180 interp error 1e-4 clad mesh size 0.8 align true -> max ~0.0035 difference (1637 seconds)
# tag '10' core res 50 clad red 150 interp error 1e-4 align true  -> max ~0.003 difference (1386 seconds)
# tag '11' core res 50 clad red 150 interp error 5e-5
# tag '12' fixed degen [[1,2],[3,4]] interp error 5e-4 -> similar results to 11 (863 seconds)
# tag '13' fixed degen, interp error 2e-4, clad res 180 core res 60 clad mesh size 0.9
# tag '14' interp error 2e-4, clad res 180 core res 60 clad mesh size 0.9
# tag '15' interp error 5e-5 dz0 0.5 clad mesh size 0.8 core mesh size 0.04 degen correction on -> one port has flipped sign ?? WHY
# tag '16' core res 80 clad res 200 clad mesh size 0.8 core mesh size 0.04 interp error 2e-5 degen correction on
# tag '17' like '16' but core mesh size -> 0.03
# tag '18' like '17' but no fixed degen
# tag '19' back to 20k length