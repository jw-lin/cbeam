import numpy as np
from wavesolve.fe_solver import solve_waveguide,get_eff_index,construct_AB,solve_sparse,plot_eigenvector
from optics import waveguide
from scipy.interpolate import UnivariateSpline

class prop:
    def __init__(self,wl,wvg:waveguide,Nmax):
        self.wvg = wvg
        self.wl = wl
        self.Nmax = Nmax
        self.k = 2*np.pi/wl
    
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

    def compute_coupling_matrix(self,z,dz):
        isect_mesh,isect_dict = self.wvg.make_intersection_mesh(z,dz)

        # current position
        A,B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        w,v,N = solve_sparse(A,B,isect_mesh,self.wl,isect_dict,num_modes=self.Nmax)

        # next position
        isect_dict = self.wvg.advance_IOR(isect_dict)
        _A,_B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        _w,_v,_N  = solve_sparse(_A,_B,isect_mesh,self.wl,isect_dict,num_modes=self.Nmax)
        #print(w)

        #for vec in v:
        #    plot_eigenvector(isect_mesh,vec,show=True)
        for i in range(v.shape[0]):
            if np.sum(np.abs(v[i]-_v[i])) < np.sum(np.abs(v[i]+_v[i])):
                continue
            else:
                _v[i]*=-1
        dvdz = v-_v

        #plot_eigenvector(isect_mesh,v[1],False,False)
        #plot_eigenvector(isect_mesh,_v[1],False,False)
        #plot_eigenvector(isect_mesh,dvdz[1],False,False)
        coupling_matrix = (B.dot(dvdz.T)).T.dot(v.T)
        return coupling_matrix

        