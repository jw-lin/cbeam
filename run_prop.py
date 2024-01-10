import numpy as np
import optics
import matplotlib.pyplot as plt

### how to run ###

# 1. create a waveguide using optics.py
# 2. initialize a propagator object with the waveguide
# 3. run prop_setup() to compute everything you need to model the lantern propagation
# 4. alternatively, load an existing set of computation results with load()
# 5. run make_interp_funcs() to set up the interpolation functions
# 6. run propagate()

# example below

### lantern params ###

wl = 1.55
taper_factor = 8
rcore = 2.2/taper_factor
rclad = 10
rjack = 30
z_ex = 20000

# ms_rcores = np.array([7.35/2,8.5/2,8.5/2,9.6/2,9.6/2,10.7/2])/taper_factor

nclad = 1.444
ncore = nclad + 8.8e-3
njack = nclad - 5.5e-3

degen = [[1,2],[3,4]]

t = 2*np.pi/5
initial_offset = rclad*2/3
xpos_i = [0,initial_offset,initial_offset*np.cos(t),initial_offset*np.cos(2*t),initial_offset*np.cos(3*t),initial_offset*np.cos(4*t)]
ypos_i = [0,0,initial_offset*np.sin(t),initial_offset*np.sin(2*t),initial_offset*np.sin(3*t),initial_offset*np.sin(4*t)]

core_pos = np.array([xpos_i,ypos_i]).T

# 1. create a waveguide (standard lantern()
lant = optics.photonic_lantern(core_pos,[rcore]*6,rclad,rjack,[ncore]*6,nclad,njack,z_ex,taper_factor,core_res=50,clad_res=150,jack_res=30,clad_mesh_size=1.0)

tag = "12" # identifier for this computation
import propagator

# 2. initialize the propagator
adprop = propagator.prop(wl,lant,6)

# 3. run prop_setup()
zs,tapervals,coupling_mats,neffs,vs,mesh = adprop.prop_setup(0,z_ex,save=True,tag=tag,max_interp_error=5e-4,fixed_degen=degen)

# 4. load the results of prop_setup() from local
adprop.load(tag=tag)


for j in range(6): # plotting cross-coupling matrix
    for i in range(j):
        plt.plot(adprop.za,adprop.cmat[:,i,j],label=str(i)+str(j))
for z in adprop.za:
    plt.axvline(x=z,alpha=0.1,color='k',zorder=-100)
plt.legend(loc='best')
plt.title("cross-coupling matrix")
plt.show()

from wavesolve.fe_solver import plot_eigenvector
w = np.power(adprop.neff,2)*adprop.k**2

for i in range(5): # looking at differences in eigenvalue (k^2 neff^2)
    plt.semilogy(adprop.za, w[:,i]-w[:,i+1],label=str(i)+str(i+1))
    
plt.semilogy(adprop.za,w[:,0]-w[:,-1],color='k')
plt.axhline(y=1e-4,color='k',ls='dashed')
plt.legend(loc='best')
plt.title("eigenvalue differences")
plt.show()

# 5. make interpolation functions
adprop.make_interp_funcs()

u0 = np.array([1,0,0,0,0,0],dtype=np.complex128) # launch LP01

# 6. propagate
z,u,uf,v = adprop.propagate(u0,z_ex)

print("final mode amplitudes: ")
print(np.abs(uf))
print(np.angle(uf))


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

mesh = adprop.mesh
print("final modes: ")
fig,axs = plt.subplots(2,3,sharex=True,sharey=True)
plot_eigenvector(mesh,adprop.vs[:,0,-1],ax=axs[0,0],show=False)
plot_eigenvector(mesh,adprop.vs[:,1,-1],ax=axs[0,1],show=False)
plot_eigenvector(mesh,adprop.vs[:,2,-1],ax=axs[0,2],show=False)
plot_eigenvector(mesh,adprop.vs[:,3,-1],ax=axs[1,0],show=False)
plot_eigenvector(mesh,adprop.vs[:,4,-1],ax=axs[1,1],show=False)
plot_eigenvector(mesh,adprop.vs[:,5,-1],ax=axs[1,2],show=False)
plt.show()

print("final field: ")
fig,axs = plt.subplots(1,3,sharey=True)
_v = np.sum(adprop.vs[:,:,-1]*uf[None,:],axis=1)
plot_eigenvector(mesh,np.real(_v),ax=axs[0],show=False)
plot_eigenvector(mesh,np.imag(_v),ax=axs[1],show=False)
plot_eigenvector(mesh,np.abs(_v),ax=axs[2],show=False)
axs[0].set_title("real part")
axs[1].set_title("imag part")
axs[2].set_title("norm")
plt.show()

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