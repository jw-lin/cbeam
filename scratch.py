import pygmsh
import numpy as np
from wavesolve.mesher import plot_mesh,lantern_mesh_6PL,circ_points
import optics
from wavesolve.fe_solver import construct_AB

import matplotlib.pyplot as plt

"""
with pygmsh.occ.Geometry() as geom:
    circ1 = geom.add_polygon(circ_points(5,16))
    circ2 = geom.add_polygon(circ_points(5,16,(1,0)))
    #circ3 = geom.add_polygon(circ_points(5,16,(15,0)))
    #circ4 = geom.add_polygon(circ_points(5,16,(15.5,0)))
    
    out = optics.boolean_fragment(geom,circ1,circ2)

    for i,frag in enumerate(out):
        geom.add_physical(frag,str(i))
    
    m = geom.generate_mesh(2,2,6)
    _dict = {"0":0,"1":1,"2":2,"3":0,"4":1,"5":2}
    plot_mesh(m,_dict)
"""

wl = 1.55
taper_factor = 8
rcore = 2.2/taper_factor
rclad = 10
rjack = 30
z_ex = 30000

ms_rcores = np.array([2.2,2.5,2.5,2.8,2.8,3.1])/taper_factor

nclad = 1.444
ncore = nclad + 8.8e-3 # contrast set for ~7.5 um MFD at lambda = 1.55 um
njack = nclad - 5.5e-3

t = 2*np.pi/5
initial_offset = rclad*2/3
xpos_i = [0,initial_offset,initial_offset*np.cos(t),initial_offset*np.cos(2*t),initial_offset*np.cos(3*t),initial_offset*np.cos(4*t)]
ypos_i = [0,0,initial_offset*np.sin(t),initial_offset*np.sin(2*t),initial_offset*np.sin(3*t),initial_offset*np.sin(4*t)]

core_pos = np.array([xpos_i,ypos_i]).T

lant = optics.linear_lantern(core_pos,ms_rcores,rclad,rjack,[ncore]*6,nclad,njack,z_ex,taper_factor,12)
#lant = optics.linear_lantern(core_pos,[rcore]*6,rclad,rjack,[ncore]*6,nclad,njack,z_ex,taper_factor,12)
#lant.update(0)
#dict = lant.assign_IOR()

#_m,_dict = lant.make_intersection_mesh(0,50)

#_dict = {"core0":1,"jacket":0,"core1":2,"cladding1":1,"core01":2,"cladding01":1,"_cladding":1}

#print(_dict)

#plot_mesh(_m,_dict)

#lant.advance_IOR(_dict)
#plot_mesh(_m,_dict)

# linearity test
import propagator
adprop = propagator.prop(wl,lant,6)

dzs = np.linspace(1,10,30)
coupling_mats = []
for dz in dzs:
    coupling_mat = adprop.compute_coupling_matrix(0,dz)
    coupling_mats.append(coupling_mat)

coupling_mats = np.array(coupling_mats)

for i in range(coupling_mats.shape[1]):
    for j in range(coupling_mats.shape[2]):
        plt.plot(dzs,np.abs(coupling_mats[:,i,j]))

plt.show()

"""

z_arr = np.linspace(0,z_ex,60)
phase_delay_funcs = adprop.get_phase_funcs(z_arr)

z_arr_hires = np.linspace(0,z_ex,200)


plt.plot(np.cos(phase_delay_funcs[0](z_arr_hires)-phase_delay_funcs[1](z_arr_hires)),label="01")

plt.plot(np.cos(phase_delay_funcs[0](z_arr_hires)-phase_delay_funcs[-1](z_arr_hires)),label="05")
plt.plot(np.cos(phase_delay_funcs[1](z_arr_hires)-phase_delay_funcs[2](z_arr_hires)),label="12")
plt.show()

#mat = adprop.compute_coupling_matrix(50,10)
#plt.imshow(mat)
#plt.show()
"""

"""
m = lant.make_mesh()



import propagator


#lant.update(z_ex)
#m = lant.make_mesh()
#lant.plot_mesh(m)
#from wavesolve.fe_solver import solve_waveguide
#w,v,N = solve_waveguide(m,wl,dict,plot=True,sparse=True)
#print(N)


za = np.concatenate((np.linspace(0,z_ex/10,20),np.linspace(z_ex/10,z_ex,10)))
adprop = propagator.prop(wl,lant)
betas = adprop.get_prop_constants(za,sparse=True)

np.save('betas',betas)


plt.xlabel(r"$z$")
plt.ylabel(r"$\beta$")
plt.axhline(y=nclad,ls='dashed',label='cladding')
plt.axhline(y=ncore,ls='dotted',label='core')
plt.axhline(y=njack,ls='-.',label="jacket")
for _b in betas.T:
    plt.plot(za,_b)

plt.legend(loc='best',frameon=False)
plt.show()
"""