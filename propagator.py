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
        self.last_v_int = None # store the last integral vals computed
                           # the signs of the next eigenvectors are
                           # set to minimize the difference
        self.last_cross_mat = None
    
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

    def adjust_sign_2(self,v,set=False):
        v_int = np.sum(v,axis=1)

        if self.last_v_int is None:
            self.last_v_int = v_int
            return

        flipmask = np.abs(v_int - self.last_v_int) > np.abs(-v_int - self.last_v_int)
        v[flipmask,:] *= -1
        v_int[flipmask] *= -1
        if set:
            self.last_v_int = v_int
            
    def compute_coupling_matrix(self,z,dz,plot=False):
        isect_mesh,isect_dict = self.wvg.make_intersection_mesh(z,dz)

        # current position
        A,B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        w,v,N = solve_sparse(A,B,isect_mesh,self.wl,isect_dict,num_modes=self.Nmax)

        # next position
        isect_dict = self.wvg.advance_IOR(isect_dict)
        _A,_B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        _w,_v,_N  = solve_sparse(_A,_B,isect_mesh,self.wl,isect_dict,num_modes=self.Nmax)

        #self.adjust_sign(B,v)
        #self.adjust_sign(_B,_v)
        #self.adjust_sign_2(_v)
        self.adjust_sign_2(v,set=True)
        self.adjust_sign_2(_v)
        #then adjust _v
        #_v *= np.where((np.abs(np.sum(_v-v,axis=1))  > np.abs(np.sum(-_v-v,axis=1))),-1,1)
        

        dvdz = (_v-v)/dz

        if plot:
            fig,axs = plt.subplots(1,2,sharey=True,figsize=(10,4))
            plot_eigenvector(isect_mesh,v[4],show=False,ax=axs[1])
            plot_eigenvector(isect_mesh,v[3],show=False,ax=axs[0])
            plt.show()

        vavg = (v+_v)/2

        coupling_matrix = (B.dot(dvdz.T)).T.dot(vavg.T) # should try to exploit symmetry, cut calcs by ~2
        
        if self.last_cross_mat is None:
            self.last_cross_mat = coupling_matrix
            return coupling_matrix
        
        mask = np.abs(coupling_matrix - self.last_cross_mat) > np.abs(-coupling_matrix - self.last_cross_mat) 
        coupling_matrix[mask] *= -1
        self.last_cross_mat = coupling_matrix
        return coupling_matrix
    
    def compute_overlap_matrix(self,z,dz):
        isect_mesh,isect_dict = self.wvg.make_intersection_mesh(z,dz)

        # current position
        A,B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        w,v,N = solve_sparse(A,B,isect_mesh,self.wl,isect_dict,num_modes=self.Nmax)

        # next position
        isect_dict = self.wvg.advance_IOR(isect_dict)
        _A,_B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        _w,_v,_N  = solve_sparse(_A,_B,isect_mesh,self.wl,isect_dict,num_modes=self.Nmax)

        self.adjust_sign(_B,_v)
        self.adjust_sign(B,v,set=True)

        mat = (B.dot(_v.T)).T.dot(v.T) 
        return mat
    
    def interpolate(self,v,inmesh,outmesh):
        """ translate v, sampled on inmesh, to outmesh """

        pass