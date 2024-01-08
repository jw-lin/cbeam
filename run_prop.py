import numpy as np
from wavesolve.mesher import plot_mesh,lantern_mesh_6PL,circ_points
import optics
from wavesolve.fe_solver import construct_AB,isinside
from wavesolve.shape_funcs import apply_affine_transform,evaluate_basis_funcs
import matplotlib.pyplot as plt

wl = 1.55
taper_factor = 8
rcore = 2.2/taper_factor
rclad = 10
rjack = 30
z_ex = 20000

ms_rcores = np.array([7.35/2,8.5/2,8.5/2,9.6/2,9.6/2,10.7/2])/taper_factor

nclad = 1.444
ncore = nclad + 8.8e-3
njack = nclad - 5.5e-3

t = 2*np.pi/5
initial_offset = rclad*2/3
xpos_i = [0,initial_offset,initial_offset*np.cos(t),initial_offset*np.cos(2*t),initial_offset*np.cos(3*t),initial_offset*np.cos(4*t)]
ypos_i = [0,0,initial_offset*np.sin(t),initial_offset*np.sin(2*t),initial_offset*np.sin(3*t),initial_offset*np.sin(4*t)]

core_pos = np.array([xpos_i,ypos_i]).T

lant = optics.linear_lantern2(core_pos,[rcore]*6,rclad,rjack,[ncore]*6,nclad,njack,z_ex,taper_factor,core_res=60,clad_res=150,jack_res=30,clad_mesh_size=1.)
# std0 core res 50 clad res 150 jack res 30 clad mesh size 1 interp error 5e-4 : 1588 seconds
# std1 core res 40 -> output is not fully symmetrcic : 824 seconds
# std2 core res 50 clad res 120 -> output is not fully symmetric : 796 seconds
# std3 same as std0 i think ? : 1450 seconds
# std4 interp error 1e-3 -> output is not fully symmetric (worst so far): 1223 seconds
# std5 core res 60 : 1139 seconds -> max ~0.01 difference in power between outer ports
# std6 core res 60 interp error 2.5e-4 : 1512 seconds -> max ~0.005 difference

import propagator
adprop = propagator.prop(wl,lant,6)

#zs,coupling_mats,betas,vi,vf,mesh = adprop.prop_setup6(0,20000,save=True,tag="std6",max_interp_error=2.5e-4)

adprop.load(tag="std0")
plt.plot(adprop.za,adprop.beta[:,0][:,None]-adprop.beta[:,1:])
for z in adprop.za:
    plt.axvline(x=z,alpha=0.1,color='k')
plt.show()

plt.plot(adprop.za,adprop.beta[:,:-1]-adprop.beta[:,1:])
plt.show()


for j in range(6):
    for i in range(j):
        plt.plot(adprop.za,adprop.cmat[:,i,j],label=str(i)+str(j))
for z in adprop.za:
    plt.axvline(x=z,alpha=0.1,color='k',zorder=-100)
plt.legend(loc='best')
plt.show()

from wavesolve.fe_solver import plot_eigenvector
w = np.power(adprop.beta,2)*adprop.k**2

for i in range(5):
    plt.semilogy(adprop.za, w[:,i]-w[:,i+1],label=str(i)+str(i+1))
    
plt.semilogy(adprop.za,w[:,0]-w[:,-1],color='k')
plt.axhline(y=1e-4,color='k',ls='dashed')
plt.legend(loc='best')
plt.show()

adprop.make_interp_funcs()

u0 = np.array([1,0,0,0,0,0],dtype=np.complex128)

z,u,uf,v = adprop.propagate(u0,20000)
print("final mode amplitudes: ")
print(np.abs(uf))
print(np.angle(uf))
for i in range(6):
    plt.plot(z,np.power(np.abs(u[i]),2),label='mode '+str(i))

plt.plot(z,np.sum(np.power(np.abs(u),2),axis=0),color='k',zorder=-100,label="total power",ls='dashed')
plt.title('LP01 propagation - std PL')

plt.xlabel(r'$z$ (um)')
plt.ylabel("power")

plt.legend(loc='best')
plt.show()

fig,axs = plt.subplots(1,2)
for i in range(6):
    axs[0].plot(z,np.real(u[i]),label='mode '+str(i))
    axs[1].plot(z,np.imag(u[i]),label='mode '+str(i))
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
plt.show()
#mesh = lant.make_mesh_bndry_ref(size_scale_fac=0.5,min_mesh_size=0.2,max_mesh_size=10.,_power=1)
#mesh = lant.make_mesh_bndry_ref(size_scale_fac=1.,min_mesh_size=0.3,max_mesh_size=10.,_power=1)

mesh = adprop.mesh
print("final modes: ")
fig,axs = plt.subplots(2,3,sharex=True,sharey=True)
plot_eigenvector(mesh,adprop.vf[0],ax=axs[0,0],show=False)
plot_eigenvector(mesh,adprop.vf[1],ax=axs[0,1],show=False)
plot_eigenvector(mesh,adprop.vf[2],ax=axs[0,2],show=False)
plot_eigenvector(mesh,adprop.vf[3],ax=axs[1,0],show=False)
plot_eigenvector(mesh,adprop.vf[4],ax=axs[1,1],show=False)
plot_eigenvector(mesh,adprop.vf[5],ax=axs[1,2],show=False)
plt.show()

print("final field: ")
fig,axs = plt.subplots(1,3,sharey=True)
plot_eigenvector(mesh,np.real(v),ax=axs[0],show=False)
plot_eigenvector(mesh,np.imag(v),ax=axs[1],show=False)
plot_eigenvector(mesh,np.abs(v),ax=axs[2],show=False)
axs[0].set_title("real part")
axs[1].set_title("imag part")
axs[2].set_title("norm")
plt.show()


