import numpy as np
from wavesolve.fe_solver import solve_waveguide,get_eff_index,construct_AB,solve_sparse,plot_eigenvector
from optics import waveguide
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
class prop:
    def __init__(self,wl,wvg:waveguide,Nmax):
        self.wvg = wvg
        self.wl = wl
        self.Nmax = Nmax
        self.k = 2*np.pi/wl
        self.signs = None
    
    def get_prop_constants(self,z_arr,sparse=True):
        betas = []
        for z in z_arr:
            self.wvg.update(z)
            IOR_dict = self.wvg.assign_IOR()
            mesh = self.wvg.make_mesh()
            w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=sparse,Nmax=self.Nmax)
            betas.append(get_eff_index(self.wl,w))
        betas = np.array(betas)
        return betas

    def get_phase_funcs(self,z_arr,sparse=True):
        betas = self.get_prop_constants(z_arr,sparse)
        betafuncs = [UnivariateSpline(z_arr,beta,s=0) for beta in betas.T]
        betafunc_ints = [func.antiderivative() for func in betafuncs]
        def _pfunc(func):
            def _inner_(z):
                return func(z)-func(0)
            return _inner_
        out = [_pfunc(bfunc_int) for bfunc_int in betafunc_ints]

        return out
    
    def get_init_sign(self):

        self.wvg.update(0)
        mesh = self.wvg.make_mesh()
        self.wvg.assign_IOR()
        A,B = construct_AB(mesh,self.wvg.IOR_dict,self.k,sparse=True)
        w,v,N = solve_sparse(A,B,mesh,self.wl,self.wvg.IOR_dict,num_modes=self.Nmax)
        signs = np.sign(np.sum(B.dot(v.T),axis=0))
        self.signs = signs
        return
    
    def adjust_sign(self,B,v):
        assert self.signs is not None, "run get_init_sign() first"
        signs = np.sign(np.sum(B.dot(v.T),axis=0))
        v *= (signs * self.signs)[:,None]
        return

    def compute_coupling_matrix(self,z,dz,plot=False):
        isect_mesh,isect_dict = self.wvg.make_intersection_mesh(z,dz)

        # current position
        A,B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        w,v,N = solve_sparse(A,B,isect_mesh,self.wl,isect_dict,num_modes=self.Nmax)

        # next position
        isect_dict = self.wvg.advance_IOR(isect_dict)
        _A,_B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        _w,_v,_N  = solve_sparse(_A,_B,isect_mesh,self.wl,isect_dict,num_modes=self.Nmax)
        #print(w)

        # make signs consistent
        self.adjust_sign(B,v)
        self.adjust_sign(_B,_v)

        dvdz = (_v-v)/dz

        if plot:
            fig,axs = plt.subplots(1,3,sharey=True,figsize=(12,4))
            plot_eigenvector(isect_mesh,dvdz[2],show=False,ax=axs[2])
            plot_eigenvector(isect_mesh,v[2],show=False,ax=axs[0])
            plot_eigenvector(isect_mesh,_v[2],show=False,ax=axs[1])
            plt.show()

        vavg = (v+_v)/2

        coupling_matrix = (B.dot(dvdz.T)).T.dot(vavg.T) # should try to exploit symmetry, cut calcs by ~2
        return coupling_matrix

        