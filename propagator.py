import numpy as np
from wavesolve.fe_solver import solve_waveguide,get_eff_index,construct_AB,solve_sparse,plot_eigenvector,isinside
from optics import waveguide
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import BVHtree
from scipy.sparse.linalg import lobpcg
import copy
from scipy.integrate import solve_ivp
from progress.bar import Bar
from wavesolve.mesher import plot_mesh
from scipy.linalg import norm
import time
import meshio
from itertools import permutations

def solve_sparse_lobpcg(A,B,x0):
    """ sparse solver for generalized eigenvalue problem Ax = lambda Bx with initial guess x0 """
    w,v = lobpcg(A,x0,B,largest=True,tol=1e-6)
    return w,v.T

class prop:
    def __init__(self,wl,wvg:waveguide,Nmax):
        self.wvg = wvg
        self.wl = wl
        self.Nmax = Nmax
        self.k = 2*np.pi/wl

        self.cmat = None
        self.beta = None
        self.vi = None
        self.vf = None

        self.cmat_funcs = None
        self.beta_funcs = None
    
    def get_prop_constants(self,z_arr,sparse=True):
        betas = []
        self.wvg.update(z_arr[0])
        mesh = self.wvg.make_mesh()
        IOR_dict = self.wvg.assign_IOR()
        for i,z in enumerate(z_arr[1:]):
            scale_fac = self.wvg.taper_func(z)/self.wvg.taper_func(z_arr[i])
            mesh.points *= scale_fac
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
    
    def make_sign_consistent_same_mesh(self,v,_v):
        flip_mask = np.sum(np.abs(v-_v),axis=1) > np.sum(np.abs(v+_v),axis=1)
        _v[flip_mask] *= -1
        return flip_mask
    
    def est_cross_term(self,B,v,_v,dz,vm=None):
        dvdz = (_v-v)/dz
        if vm is None:
            vavg = (v+_v)/2.
        else:
            vavg = vm
        return (B.dot(dvdz.T)).T.dot(vavg.T)

    def find_degenerate_groups(self,betas,min_dif=1e-4):
        groups = []
        idxs = []
        for i in range(len(betas)):
            group = []
            if i in idxs:
                continue
            for j in range(i+1,len(betas)):
                if betas[i] - betas[j] < min_dif:
                    if not group:
                        group.append(i)
                        idxs.append(i)
                    group.append(j)
                    idxs.append(j)
            if group:
                groups.append(group)
        return groups

    def find_degenerate_groups_2(self,betas,min_dif=1e-4):
        """ any eigenvalues w/in min_dif are treated as degenerate. a chain of eigenvalues
            with adjacent differences all less than min_dif are treated as a single 
            degenerate group.
        """
        groups = []
        idxs = []
        for i in range(len(betas)):
            if i in idxs:
                continue
            group = [i]
            for j in range(i+1,len(betas)):
                if betas[group][-1] - betas[j] < min_dif:
                    group.append(j)
                    idxs.append(j) 
            groups.append(group)
        return groups

    #sln from https://math.stackexchange.com/questions/2399929/how-to-get-the-same-eigenvectors-of-a-degenerate-eigenvalue-as-the-matrix-evolve
    def correct_degeneracy(self,group,v,_v,q=None):
        if q is None:
            coeff_mat = np.dot(v[group,:],_v[group,:].T)
            u,s,vh = np.linalg.svd(coeff_mat)
            q = np.dot(vh.T,u.T)
        _vq = np.dot(_v[group,:].T,q)
        for i,g in enumerate(group):
            _v[g] = _vq[:,i]
        return v,_v,q

    def update_degen_groups(self,betas,min_dif,last_degen_groups):
        """ this func assumes the degenerate groups can only become larger
            (so the eigenvalues of the waveguide are converging with propagation.)
            i believe this is the only way for the problem to be well defined.
        """
        degen_groups = copy.deepcopy(last_degen_groups)
        degen_groups_flat = [x for gr in degen_groups for x in gr]

        # look for adjacent beta values not already in a group and fit them together
        for i in range(1,len(betas)):
            if i in degen_groups_flat or i-1 in degen_groups_flat:
                continue
            if betas[i-1]-betas[i] < min_dif:
                # find the correct index to insert at
                if len(degen_groups) == []:
                    degen_groups.append([i-1,i])
                else:
                    for j in range(len(degen_groups)):
                        if i < degen_groups[j][0]:
                            degen_groups.insert(j,[i-1,i])
                degen_groups_flat.append(i-1)
                degen_groups_flat.append(i)

        # look at beta values not already in a group and try to fit them into each degen group
        for i in range(len(betas)):
            b = betas[i]
            if i in degen_groups_flat:
                continue
            elif len(degen_groups) == 0:
                continue
            for gr in degen_groups:
                if i < gr[0] and b-betas[gr[-1]] < min_dif:
                    gr.insert(0,i)
                    break
                elif i > gr[-1] and betas[gr[0]] - b < min_dif:
                    gr.append(i)
                    break

        # look at groups and try to combine if the total difference is less than min_dif
        if len(degen_groups)>1:
            i = 1
            while i < len(degen_groups):
                if betas[degen_groups[i-1][0]] - betas[degen_groups[i][-1]] < min_dif:
                    biggroup = degen_groups[i-1]+degen_groups[i]
                    degen_groups[i] = biggroup
                    degen_groups.pop(i-1)
                else:
                    i+=1
        return degen_groups


    def avg_degen_beta(self,groups,betas):
        for gr in groups:
            betas[gr] = np.mean(betas[gr]) 
        return betas
        
    def compute_cmats(self,z_arr,w_thresh=3e-3):
        cmats = np.zeros((z_arr.shape[0],self.Nmax,self.Nmax))
        dz = z_arr[1]-z_arr[0]
        z = z_arr[0]

        # step 0
        isect_mesh,isect_dict = self.wvg.make_intersection_mesh(z,dz)

        # current position
        A,B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        w,v,N = solve_sparse(A,B,isect_mesh,self.k,isect_dict,num_modes=self.Nmax)

        # next position
        isect_dict = self.wvg.advance_IOR(isect_dict)
        _A,_B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
        _w,_v,_N  = solve_sparse(_A,_B,isect_mesh,self.k,isect_dict,num_modes=self.Nmax) 
        
        degen_groups = self.find_degenerate_groups(w,w_thresh)

        for gr in degen_groups:
            v,_v,q = self.correct_degeneracy(gr,v,_v)
        
        self.make_sign_consistent_same_mesh(v,_v)
    
        cmats[0,:] = self.est_cross_term(B,v,_v,dz)

        for j,z in enumerate(z_arr[1:]):
            i = j+1
            dz = z_arr[i]-z_arr[i-1]
            z = z_arr[i]

            next_isect_mesh,next_isect_dict = self.wvg.make_intersection_mesh(z,dz)

            tripoints_all = isect_mesh.points[isect_mesh.cells[1].data,:2]
            meshtree = BVHtree.create_tree(tripoints_all[:,:3,:])
                    
            # interpolate old v onto new mesh
            vlast = np.empty((_v.shape[0],next_isect_mesh.points.shape[0]),dtype=np.float64)
            triidxs,interpweights = BVHtree.get_idxs_and_weights(next_isect_mesh.points,meshtree)
            for k,vec in enumerate(_v):
                vlast[k,:] = fast_interpolate(vec,isect_mesh,triidxs,interpweights)

            A,B = construct_AB(next_isect_mesh,next_isect_dict,self.k,sparse=True)
            w,v,N = solve_sparse(A,B,next_isect_mesh,self.k,next_isect_dict,num_modes=self.Nmax)

            # next position
            next_isect_dict = self.wvg.advance_IOR(next_isect_dict)
            _A,_B = construct_AB(next_isect_mesh,next_isect_dict,self.k,sparse=True)
            _w,_v,_N  = solve_sparse(_A,_B,next_isect_mesh,self.k,next_isect_dict,num_modes=self.Nmax)

            #degen_groups = self.find_degenerate_groups(w,w_thresh)
            #print(degen_groups)
            #print(_w)

            for gr in degen_groups:
                vlast,v,q = self.correct_degeneracy(gr,vlast,v)
                v,_v,_q = self.correct_degeneracy(gr,v,_v)

            self.make_sign_consistent_same_mesh(vlast,v)
            self.make_sign_consistent_same_mesh(v,_v)

            plot = True
            if plot:
                fig,axs = plt.subplots(3,6,sharey=True)
                plot_eigenvector(next_isect_mesh,v[0],ax=axs[0,0],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,v[1],ax=axs[0,1],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,v[2],ax=axs[0,2],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,v[3],ax=axs[0,3],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,v[4],ax=axs[0,4],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,v[5],ax=axs[0,5],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[0],ax=axs[1,0],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[1],ax=axs[1,1],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[2],ax=axs[1,2],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[3],ax=axs[1,3],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[4],ax=axs[1,4],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[5],ax=axs[1,5],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[0]-v[0],ax=axs[2,0],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[1]-v[1],ax=axs[2,1],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[2]-v[2],ax=axs[2,2],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[3]-v[3],ax=axs[2,3],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[4]-v[4],ax=axs[2,4],show=False,show_mesh=False)
                plot_eigenvector(next_isect_mesh,_v[5]-v[5],ax=axs[2,5],show=False,show_mesh=False)
                plt.show()
            
            cmats[i,:,:] = self.est_cross_term(B,v,_v,dz)  
            isect_mesh = next_isect_mesh
        return cmats                                 

    def prop_setup(self,z_arr,w_thresh=3e-4,initfromzero=True,fixed_degen=None,save=False):
        cmats = np.zeros((z_arr.shape[0],self.Nmax,self.Nmax))
        betas = np.zeros((z_arr.shape[0],self.Nmax))
        dz = z_arr[1]-z_arr[0]
        #dz = 50
        z = z_arr[0]

        # step 0
        if initfromzero:
            self.wvg.update(0)
            mesh = self.wvg.make_mesh()
            _mesh = self.wvg.make_mesh() # copy
            dict = self.wvg.assign_IOR()
            scale_fac = self.wvg.taper_func(z)/self.wvg.taper_func(0)
            mesh.points *= scale_fac
            _mesh.points *= scale_fac
        else:
            self.wvg.update(z)
            mesh = self.wvg.make_mesh()
            _mesh = self.wvg.make_mesh() # copy
            dict = self.wvg.assign_IOR()

        # current position
        w,v,N = solve_waveguide(mesh,self.wl,dict,sparse=True,Nmax=self.Nmax)

        v_init = np.copy(v)
    
        if fixed_degen is None:
            degen_groups = self.find_degenerate_groups(w,w_thresh)
        else:
            degen_groups = fixed_degen

        # put points into tree
        tripoints = mesh.points[mesh.cells[1].data,:2]
        meshtree = BVHtree.create_tree(tripoints[:,:3,:])
        betas[0,:] = get_eff_index(self.wl,w)
        for i in range(1,len(z_arr)):
            z = z_arr[i]
            dz = z - z_arr[i-1]
            scale_fac = self.wvg.taper_func(z)/self.wvg.taper_func(z_arr[i-1])

            # rescale mesh points
            _mesh.points *= scale_fac

            # put points into tree
            _tripoints = _mesh.points[_mesh.cells[1].data,:2]
            _meshtree = BVHtree.create_tree(_tripoints[:,:3,:])

            # solve on scaled mesh
            _w,_v,_N = solve_waveguide(_mesh,self.wl,dict,sparse=True,Nmax=self.Nmax)
            
            # create an intersection mesh
            isect_mesh,isect_dict = self.wvg.make_intersection_mesh(z,dz)
            
            # interpolate both the last and current vectors onto this mesh
            vi = np.empty((self.Nmax,isect_mesh.points.shape[0]),dtype=np.float64)

            triidxs,interpweights = BVHtree.get_idxs_and_weights(isect_mesh.points,meshtree)
            for k,vec in enumerate(v):
                vi[k,:] = fast_interpolate(vec,mesh,triidxs,interpweights)

            _vi = np.empty((self.Nmax,isect_mesh.points.shape[0]),dtype=np.float64)

            _triidxs,_interpweights = BVHtree.get_idxs_and_weights(isect_mesh.points,_meshtree)
            for k,vec in enumerate(_v):
                _vi[k,:] = fast_interpolate(vec,_mesh,_triidxs,_interpweights)

            # correct for sign
            flip_mask = self.make_sign_consistent_same_mesh(vi,_vi)
            _v[flip_mask,:]*=-1

            #_degen_groups = self.find_degenerate_groups(_w,w_thresh)
            if fixed_degen is None:
                _degen_groups = self.update_degen_groups(_w,w_thresh,degen_groups)
            else:
                _degen_groups = degen_groups

            # correct degeneracy - forward direction
            for gr in _degen_groups:
                vi,_vi,q = self.correct_degeneracy(gr,vi,_vi)
                v,_v,q = self.correct_degeneracy(gr,v,_v,q)
            
            # correct degeneracy - backward direction
            #for gr in degen_groups:
            #    _vi,vi,q = self.correct_degeneracy(gr,_vi,vi)
            #   _v,v,q = self.correct_degeneracy(gr,_v,v,q)

            A,B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)

            plot=False
            print(degen_groups)
            if plot:
                fig,axs = plt.subplots(3,6)
                for j in range(6):
                    plot_eigenvector(isect_mesh,vi[j],ax=axs[0,j],show=False)
                    plot_eigenvector(isect_mesh,_vi[j],ax=axs[1,j],show=False)
                    plot_eigenvector(isect_mesh,vi[j]-_vi[j],ax=axs[2,j],show=False)
                plt.show()

            # compute cross term
            cmats[i,:,:] = self.est_cross_term(B,vi,_vi,dz) 
            fig,axs = plt.subplots(1,2)
            axs[0].imshow(cmats[i,:,:])
            axs[1].imshow(np.abs(cmats[i,:,:]+cmats[i,:,:].T))
            plt.show()

            # save beta
            _betas = get_eff_index(self.wl,_w)
            self.avg_degen_beta(degen_groups,_betas)
            betas[i,:] = _betas[:]

            # recycle some computations
            meshtree = _meshtree
            mesh.points = np.copy(_mesh.points)
            v = _v
            w = _w 
            degen_groups = _degen_groups
        v_final = np.copy(_v)
        if save:
            np.save("cmat",cmats[1:])
            np.save("beta",betas)
            np.save("vi",v_init)
            np.save("vf",v_final)
        print("final eigenmodes: ")
        for i in range(self.Nmax):
            plot_eigenvector(_mesh,v_final[i])
        return cmats,betas,v_init,v_final   

    def prop_setup2(self,z_arr,save=False,dz0=10,fixed_degen=None):
        cmats = np.zeros((z_arr.shape[0],self.Nmax,self.Nmax))
        betas = np.zeros((z_arr.shape[0],self.Nmax))

        zi = z_arr[0]

        self.wvg.update(0)

        #always init the mesh from 0
        isect_mesh,isect_dict = self.wvg.make_intersection_mesh(-dz0/2,dz0)

        isect_mesh.points *= self.wvg.taper_func(zi)/self.wvg.taper_func(0)

        _isect_dict = isect_dict.copy()
        self.wvg.advance_IOR(_isect_dict)
        vi = None
        bar = Bar('Processing', max=len(z_arr))
        vlast = None
        _vlast = None
        avg_vlast = None
        for i,z in enumerate(z_arr):
            total_scale_fac =  self.wvg.taper_func(z)/self.wvg.taper_func(z_arr[0])
            if i>0:
                scale_fac = self.wvg.taper_func(z)/self.wvg.taper_func(z_arr[i-1])
            else:
                scale_fac = 1
            
            isect_mesh.points *= scale_fac

            # eigen solve
            w,v,N = solve_waveguide(isect_mesh,self.wl,isect_dict,sparse=True,Nmax=self.Nmax)
            _w,_v,_N = solve_waveguide(isect_mesh,self.wl,_isect_dict,sparse=True,Nmax=self.Nmax)

            # sign correction
            if vlast is None:
                self.make_sign_consistent_same_mesh(v,_v)
            else:
                self.make_sign_consistent_same_mesh(vlast,v)
                self.make_sign_consistent_same_mesh(_vlast,_v)

                # degeneracy correction
                if fixed_degen is not None:
                    for gr in fixed_degen:
                        self.correct_degeneracy(gr,vlast,v)
                        self.correct_degeneracy(gr,_vlast,_v)

            vlast = v
            _vlast = _v
            avg_vlast = 0.5*(v+_v)

            # est deriv
            A,B = construct_AB(isect_mesh,isect_dict,self.k,sparse=True)
            cmats[i,:,:] = self.est_cross_term(B,v,_v,dz0*total_scale_fac) 
    
            # save initial
            if i == 0:
                vi = np.copy(avg_vlast)

            # save beta
            betas[i,:] = get_eff_index(self.wl,0.5*(w+_w))
            print(betas[i,:])

            plot=True
            if plot:
                fig,axs = plt.subplots(3,6)
                for j in range(6):
                    sgn = 1
                    if j in [0,1,2,5]:
                        sgn=-1
                    plot_eigenvector(isect_mesh,v[j]*sgn,ax=axs[0,j],show=False)
                    plot_eigenvector(isect_mesh,_v[j]*sgn,ax=axs[1,j],show=False)
                    plot_eigenvector(isect_mesh,(v[j]-_v[j])*sgn,ax=axs[2,j],show=False)
                plt.show()
            bar.next()

        vf = np.copy(avg_vlast)
        
        if save:
            np.save("cmat",cmats)
            np.save("beta",betas)
            np.save("vi",vi)
            np.save("vf",vf)
        bar.finish()
        return cmats,betas,vi,vf  

    def prop_setup3(self,z_arr,save=False,dz0=1,tag=""):
        cmats = np.zeros((z_arr.shape[0],self.Nmax,self.Nmax))
        betas = np.zeros((z_arr.shape[0],self.Nmax))

        zi = z_arr[0]

        self.wvg.update(0)

        #always init the mesh from 0
        #mesh = self.wvg.make_mesh()
        #_mesh = self.wvg.make_mesh()
        #middle_mesh = self.wvg.make_mesh()
        mesh = self.wvg.make_mesh_bndry_ref(size_scale_fac=0.5,min_mesh_size=0.2,max_mesh_size=10.,_power=1)
        _mesh = copy.deepcopy(mesh)
        middle_mesh = copy.deepcopy(mesh)

        IOR_dict = self.wvg.assign_IOR()

        points0 = np.copy(mesh.points * self.wvg.taper_func(zi))
        
        vi = None
        bar = Bar('Processing', max=len(z_arr))
        vlast = None

        for i,z in enumerate(z_arr):
            scale_facm = self.wvg.taper_func(z-dz0/2)/self.wvg.taper_func(zi)
            scale_facp = self.wvg.taper_func(z+dz0/2)/self.wvg.taper_func(zi)
            scale_fac = self.wvg.taper_func(z)/self.wvg.taper_func(zi)

            # minus step
            mesh.points = points0*scale_facm
            w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            tripoints = mesh.points[mesh.cells[1].data,:2]
            meshtree = BVHtree.create_tree(tripoints[:,:3,:])

            # plus step
            _mesh.points = points0*scale_facp
            _w,_v,_N = solve_waveguide(_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            _tripoints = _mesh.points[_mesh.cells[1].data,:2]
            _meshtree = BVHtree.create_tree(_tripoints[:,:3,:])

            # interpolation onto middle mesh
            middle_mesh.points = points0*scale_fac
            #wm,vm,Nm = solve_waveguide(middle_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            A,B = construct_AB(middle_mesh,IOR_dict,self.k,sparse=True)

            vinterp = np.empty((self.Nmax,middle_mesh.points.shape[0]),dtype=np.float64)

            triidxs,interpweights = BVHtree.get_idxs_and_weights(middle_mesh.points,meshtree)
            for k,vec in enumerate(v):
                vinterp[k,:] = fast_interpolate(vec,mesh,triidxs,interpweights)

            _vinterp = np.empty((self.Nmax,middle_mesh.points.shape[0]),dtype=np.float64)

            _triidxs,_interpweights = BVHtree.get_idxs_and_weights(middle_mesh.points,_meshtree)
            for k,vec in enumerate(_v):
                _vinterp[k,:] = fast_interpolate(vec,_mesh,_triidxs,_interpweights)

            # sign correction
            if vlast is None:
                self.make_sign_consistent_same_mesh(vinterp,_vinterp)
            else:
                self.make_sign_consistent_same_mesh(vlast,vinterp)
                self.make_sign_consistent_same_mesh(vlast,_vinterp)

            vlast = 0.5*(vinterp+_vinterp)

            # est deriv
            #cmats[i,:,:] = 0.5* (self.est_cross_term(B,vinterp,vm,dz0/2,vm)+self.est_cross_term(B,vm,_vinterp,dz0/2,vm))
            cmats[i,:,:] = self.est_cross_term(B,vinterp,_vinterp,dz0)
            plt.imshow(cmats[i,:,:]+cmats[i,:,:].T)
            plt.show()
    
            # save initial
            if i == 0:
                vi = np.copy(vlast)

            # save beta
            betas[i,:] = get_eff_index(self.wl,0.5*(w+_w))
            #print(betas[i,:])
            plot=True
            if plot:
                fig,axs = plt.subplots(3,6)
                for j in range(6):
                    plot_eigenvector(middle_mesh,vinterp[j],ax=axs[0,j],show=False)
                    plot_eigenvector(middle_mesh,_vinterp[j],ax=axs[1,j],show=False)
                    plot_eigenvector(middle_mesh,vinterp[j]-_vinterp[j],ax=axs[2,j],show=False)
                plt.show()
            bar.next()

        vf = np.copy(vlast)
        
        if save:
            np.save("cmat"+tag,cmats)
            np.save("beta"+tag,betas)
            np.save("vi"+tag,vi)
            np.save("vf"+tag,vf)
        bar.finish()
        return cmats,betas,vi,vf  

    def prop_setup4(self,z_arr,save=False,dz0=1,tag="",degen_cor=True):
        thresh=7e-4
        cmats = np.zeros((z_arr.shape[0],self.Nmax,self.Nmax))
        betas = np.zeros((z_arr.shape[0],self.Nmax))

        zi = z_arr[0]

        self.wvg.update(0)

        #always init the mesh from 0
        #mesh = self.wvg.make_mesh_bndry_ref(size_scale_fac=0.5,min_mesh_size=0.2,max_mesh_size=10.,_power=1)
        #mesh = self.wvg.make_mesh_bndry_ref(size_scale_fac=1.,min_mesh_size=0.3,max_mesh_size=10.,_power=1)
        mesh = self.wvg.make_mesh_bndry_ref(size_scale_fac=1.,min_mesh_size=0.4,max_mesh_size=10.,_power=1,write=True)
        _mesh = copy.deepcopy(mesh)
        #middle_mesh = copy.deepcopy(mesh)
        #print("starting mesh: ")
        
        IOR_dict = self.wvg.assign_IOR()
        #plot_mesh(mesh,IOR_dict)

        points0 = np.copy(mesh.points * self.wvg.taper_func(zi))
        
        vi = None
        bar = Bar('Processing', max=len(z_arr))
        vlast = None

        for i,z in enumerate(z_arr):
            scale_facm = self.wvg.taper_func(z-dz0/2)/self.wvg.taper_func(zi)
            scale_facp = self.wvg.taper_func(z+dz0/2)/self.wvg.taper_func(zi)

            # minus step
            mesh.points = points0*scale_facm
            w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            tripoints = mesh.points[mesh.cells[1].data,:2]
            meshtree = BVHtree.create_tree(tripoints[:,:3,:])

            # plus step
            _mesh.points = points0*scale_facp
            _w,_v,_N = solve_waveguide(_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            _A,_B = construct_AB(_mesh,IOR_dict,self.k,sparse=True)

            if degen_cor:
                degen_groups = self.find_degenerate_groups_2(_w,thresh)
                print(degen_groups)

            # sign correction and degen correction
            if vlast is None:
                self.make_sign_consistent_same_mesh(v,_v)
                if degen_cor:
                    for gr in degen_groups:
                        self.correct_degeneracy(gr,v,_v)
            else:
                self.make_sign_consistent_same_mesh(vlast,v)
                self.make_sign_consistent_same_mesh(vlast,_v)
                if degen_cor:
                    for gr in degen_groups:
                        self.correct_degeneracy(gr,vlast,v)
                        self.correct_degeneracy(gr,vlast,_v)

            vlast = 0.5*(v+_v)

            # interpolation

            vinterp = np.copy(_v)

            triidxs,interpweights = BVHtree.get_idxs_and_weights(_mesh.points,meshtree)

            mask = (triidxs != -1)
            for k,vec in enumerate(v):
                vinterp[k,:][mask] = fast_interpolate(vec,mesh,triidxs[mask],interpweights[mask])

            # est deriv
            #cmats[i,:,:] = 0.5* (self.est_cross_term(B,vinterp,vm,dz0/2,vm)+self.est_cross_term(B,vm,_vinterp,dz0/2,vm))
            cmats[i,:,:] = self.est_cross_term(_B,vinterp,_v,dz0)

            # save initial
            if i == 0:
                vi = np.copy(vlast)

            # save beta
            betas[i,:] = get_eff_index(self.wl,0.5*(w+_w))
            #print(betas[i,:])

            plot=False
            if plot:
                fig,axs = plt.subplots(3,6,sharex=True,sharey=True)
                for j in range(6):
                    plot_eigenvector(_mesh,vinterp[j],ax=axs[0,j],show=False)
                    plot_eigenvector(_mesh,_v[j],ax=axs[1,j],show=False)
                    plot_eigenvector(_mesh,vinterp[j]-_v[j],ax=axs[2,j],show=False)
                plt.subplots_adjust(hspace=0,wspace=0)
                plt.show()

            bar.next()

        vf = np.copy(vlast)
        
        if save:
            np.save("cmat"+tag,cmats)
            np.save("beta"+tag,betas)
            np.save("vi"+tag,vi)
            np.save("vf"+tag,vf)
        bar.finish()
        return cmats,betas,vi,vf  

    def compute(self,z,zi,mesh,_mesh,IOR_dict,points0,dz0,betas,zs,perm=None,vlast=None,cross=False):
        scale_facm = self.wvg.taper_func(z-dz0/2)/self.wvg.taper_func(zi)
        scale_facp = self.wvg.taper_func(z+dz0/2)/self.wvg.taper_func(zi)

        # minus step
        mesh.points = points0*scale_facm
        w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
        tripoints = mesh.points[mesh.cells[1].data,:2]
        meshtree = BVHtree.create_tree(tripoints[:,:3,:])

        # plus step
        _mesh.points = points0*scale_facp
        _w,_v,_N = solve_waveguide(_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
        _A,_B = construct_AB(_mesh,IOR_dict,self.k,sparse=True)

        # make a permutation if required
        if cross and perm is not None:
            v = v[perm]
            _v = _v[perm]
            w = w[perm]
            _w = _w[perm]

        # sign correction
        if vlast is None:
            self.make_sign_consistent_same_mesh(v,_v)
        else:
            self.make_sign_consistent_same_mesh(vlast,v)
            self.make_sign_consistent_same_mesh(vlast,_v)

        # check crossing
        if cross and len(betas)>=4:
            _perm = self.check_crossing(zs[-4:],np.array(betas)[-4:,:],z,(w+_w)/2)
            if _perm is not None:
                v = v[_perm]
                _v = _v[_perm]
                w = w[_perm]
                _w = _w[_perm]
                if perm is None:
                    perm = _perm
                else:
                    print(perm,_perm)
                    perm = np.array(perm)[_perm]

        vlast = 0.5*(v+_v)

        # interpolation

        vinterp = np.copy(_v)

        triidxs,interpweights = BVHtree.get_idxs_and_weights(_mesh.points,meshtree)

        mask = (triidxs != -1)
        for k,vec in enumerate(v):
            vinterp[k,:][mask] = fast_interpolate(vec,mesh,triidxs[mask],interpweights[mask])

        # est deriv
        cmat = self.est_cross_term(_B,vinterp,_v,dz0)
        betas = get_eff_index(self.wl,0.5*(w+_w))
        return cmat,betas,vlast,perm

    def prop_setup5(self,zi,zf,max_interp_error=1e-6,save=False,dz0=1,tag="",min_zstep=1):
        """ compute the initial/final eigenmodes, eigenvalues, and cross-coupling coefficients for the 
            waveguide loaded into self.wvg. waveguide evolution wrt z is assumed to be a linear and uniform 
            scaling in x,y.

        ARGS: 
            zi: initial z coordinate (dyou can start computation in the middle of the wavguide)
            zf: final z coordinate
            max_interp_error: the z-step is chosen adaptively so that the norm of the cross-coupling 
                              matrix is less than <max_interp_error> off the value predicted by cubic 
                              spline extrapolation from the last 4 values of the cross-coupling matrix.
            save: set True to save all results to file. you can load later with load(<tag>)
            dz0: the default value to use in numerical estimation of eigenmode derivatives
            tag: the unique string which will identify the files for this computation
            min_zstep: if the adaptively chosen z-step falls below this threshhold, an exception is raised.
        RETURNS:
            za: array of z values
            cmats: array of coupling coefficient matrices through the waveguide
            betas: array of eigenvalues (effective indices) through the waveguide (yes beta is the wrong symbol for this)
            vi: initial eigenmodes computed at z = zi
            vf: final eigenmodes computed at z = zf
            mesh: the finite element mesh used for the computation. vi and vf are computed on mesh.points                   
        """

        start_time = time.time()
        zstep0 = 10

        cmats = []
        cmats_norm = []
        betas = []
        zs = []

        self.wvg.update(0)

        #always init the mesh from 0
        if save:
            meshwriteto="mesh"+tag
        else:
            meshwriteto=None
        mesh = self.wvg.make_mesh_bndry_ref(size_scale_fac=1.,min_mesh_size=0.4,max_mesh_size=10.,_power=1,writeto=meshwriteto)
        _mesh = copy.deepcopy(mesh)
        
        IOR_dict = self.wvg.assign_IOR()

        points0 = np.copy(mesh.points * self.wvg.taper_func(zi))
        
        vi = None
        vlast = None

        z = zi
        while True:
            if zstep0<min_zstep:
                print("comp. halted")
                raise Exception("error: zstep has decreased below lower limit - eigenmodes may not be varying continuously")
            dz = min(zstep0/10,dz0)
            cmat,beta,vlast_temp = self.compute(z,zi,mesh,_mesh,IOR_dict,points0,dz,vlast=vlast)
            cnorm = norm(cmat)
            if len(zs)<4:
                cmats.append(cmat)
                betas.append(beta)
                cmats_norm.append(cnorm)
                zs.append(z)
                vlast = vlast_temp
                if z == zi:
                    vi = np.copy(vlast)
                if z == zf:
                    break
                z = min(zf,z+zstep0)
                continue

            # construct spline
            spl = UnivariateSpline(zs[-4:],cmats_norm[-4:],k=3,s=0,ext=0)
            err = np.abs(spl(z)-cnorm)
            if err<max_interp_error:
                cmats.append(cmat)
                betas.append(beta)
                cmats_norm.append(cnorm)
                zs.append(z)
                vlast = vlast_temp
                if z == zf:
                    break
                # rescale zstep0
                if err<0.1*max_interp_error:
                    zstep0*=2
                z = min(zf,z+zstep0)
                print("current z: ",z)
                print("current zstep: ",zstep0)
            else:
                print(z,": tolerance not met, reducing ztep")
                print("error: ",err)
                # try again
                z -= zstep0
                zstep0/=2
                z += zstep0
                
        vf = np.copy(vlast)
        
        if save:
            np.save("cmat"+tag,np.array(cmats))
            np.save("beta"+tag,np.array(betas))
            np.save("vi"+tag,vi)
            np.save("vf"+tag,vf)
            np.save("za"+tag,np.array(zs))
        print("time elapsed: ",time.time()-start_time)
        return zs,cmats,betas,vi,vf,mesh

    def check_crossing(self,last_zs,last_betas,z,beta,thresh=1e-9):
        """ check if eigenvalues are crossing. if so, return the permuation that 
            needs to be made. otherwise, return None. """
        groups = self.find_degenerate_groups_2(last_betas[-1,:],min_dif=thresh)
        mask = [len(gr)==1 for gr in groups]
        if np.all(mask):
            return None# beta vals are well separated, we don't need to do anything.
        perms = copy.deepcopy(groups)
        for j,gr in enumerate(groups):
            if len(gr)==1:
                continue
            last_betas_gr = last_betas[:,gr]
            spline_funcs = [UnivariateSpline(last_zs,b,s=0,k=3,ext=0) for b in last_betas_gr.T]
            beta_interp = np.array([s(z) for s in spline_funcs])
            # permute beta and look for best fit
            resid = np.sum(np.power(beta[gr]-beta_interp,2))
            best_perm = None
            for i,perm in enumerate(permutations(gr)):
                perm = list(perm)
                if i == 0:
                    continue # 0 perm is the original
                _resid = np.sum(np.power(beta[perm]-beta_interp,2))
                if _resid < resid:
                    resid = _resid
                    best_perm = perm
            if best_perm is not None:
                perms[j] = best_perm
        return [p for perm in perms for p in perm] # flattened

    def prop_setup6(self,zi,zf,max_interp_error=3e-4,save=False,dz0=1,tag="",min_zstep=1,cross=False):
        """ like 5 but we try to deal with eigenvalue crossing (unsuccessfully for now)             
        """

        start_time = time.time()
        zstep0 = 10

        cmats = []
        cmats_norm = []
        betas = []
        zs = []

        self.wvg.update(0)

        #always init the mesh from 0
        if save:
            meshwriteto="mesh"+tag
        else:
            meshwriteto=None
        mesh = self.wvg.make_mesh_bndry_ref(size_scale_fac=1.,min_mesh_size=0.4,max_mesh_size=10.,_power=1,writeto=meshwriteto)
        _mesh = copy.deepcopy(mesh)
        
        IOR_dict = self.wvg.assign_IOR()

        points0 = np.copy(mesh.points * self.wvg.taper_func(zi))
        
        vi = None
        vlast = None
        perm = None

        z = zi
        while True:
            if zstep0<min_zstep:
                print("comp. halted")
                raise Exception("error: zstep has decreased below lower limit - eigenmodes may not be varying continuously")
            dz = min(zstep0/10,dz0)
            cmat,beta,vlast_temp,new_perm = self.compute(z,zi,mesh,_mesh,IOR_dict,points0,dz,betas,zs,vlast=vlast,perm=perm,cross=cross)
            cnorm = norm(cmat)
            if len(zs)<4:
                cmats.append(cmat)
                betas.append(beta)
                cmats_norm.append(cnorm)
                zs.append(z)
                vlast = vlast_temp
                if z == zi:
                    vi = np.copy(vlast)
                if z == zf:
                    break
                z = min(zf,z+zstep0)
                continue

            # construct spline
            spl = UnivariateSpline(zs[-4:],cmats_norm[-4:],k=3,s=0,ext=0)
            err = np.abs(spl(z)-cnorm)
            if err<max_interp_error:
                cmats.append(cmat)
                betas.append(beta)
                cmats_norm.append(cnorm)
                zs.append(z)
                vlast = vlast_temp
                if z == zf:
                    break
                # rescale zstep0
                if err<0.1*max_interp_error:
                    zstep0*=2
                z = min(zf,z+zstep0)
                print("current z: ",z)
                print("current zstep: ",zstep0)
                perm = new_perm
                print("current perm: ",perm)
            else:
                print(z,": tolerance not met, reducing ztep")
                print("error: ",err)                
                z -= zstep0
                zstep0/=2
                z += zstep0
                
        vf = np.copy(vlast)
        
        if save:
            np.save("cmat"+tag,np.array(cmats))
            np.save("beta"+tag,np.array(betas))
            np.save("vi"+tag,vi)
            np.save("vf"+tag,vf)
            np.save("za"+tag,np.array(zs))
        print("time elapsed: ",time.time()-start_time)
        return zs,cmats,betas,vi,vf,mesh

    def load(self,tag=""):
        self.cmat = np.load("cmat"+tag+".npy")
        self.beta = np.load("beta"+tag+".npy")
        self.vi = np.load("vi"+tag+".npy")
        self.vf = np.load("vf"+tag+".npy")
        try:
            self.za = np.load("za"+tag+".npy")
            self.mesh = meshio.read("mesh"+tag+".msh")
        except:
            pass
    
    def make_interp_funcs(self):
        za = self.za
        def make_c_func(i,j):
            assert i < j, "i must be < j in make_interp_funcs()"
            return UnivariateSpline(za,0.5*(self.cmat[:,i,j]-self.cmat[:,j,i]),ext=0,s=0)
        cmat_funcs = []
        for j in range(1,self.Nmax):
            for i in range(j):
                cmat_funcs.append(make_c_func(i,j))
        
        beta_funcs = []
        for i in range(self.Nmax):
            beta_funcs.append(UnivariateSpline(za,self.beta[:,i],s=0))
    
        self.cmat_funcs = cmat_funcs
        self.beta_funcs = beta_funcs
        self.beta_int_funcs = [beta_func.antiderivative() for beta_func in beta_funcs]
    
    def compute_cmat(self,z):
        out = np.zeros((self.Nmax,self.Nmax))
        k = 0
        for j in range(1,self.Nmax):
            for i in range(j):
                val = self.cmat_funcs[k](z)
                out[i,j] = -val
                out[j,i] = val
                k+=1
        return out
    
    def compute_beta(self,z):
        return np.array([beta(z) for beta in self.beta_funcs])
    
    def compute_int_beta(self,z):
        return np.array([betai(z) for betai in self.beta_int_funcs])

    def propagate(self,u0,zf):
        zi = self.za[0]
        def deriv(z,u):
            betas = self.compute_beta(z)
            phases = self.k * (self.compute_int_beta(z)-self.compute_int_beta(zi))
            cmat = self.compute_cmat(z)
            phase_mat = np.exp(1.j * (phases[None,:] - phases[:,None]))
            return -1./betas*np.dot(phase_mat*cmat,u*betas)
        
        sol = solve_ivp(deriv,(zi,zf),u0,'RK23')
        # multiply by phase factors
        final_phase = np.exp(1.j*self.k*np.array(self.compute_int_beta(zf)-self.compute_int_beta(zi)))
        uf = sol.y[:,-1]*final_phase
        v = np.sum(uf[:,None]*self.vf,axis=0)
        return sol.t,sol.y,uf,v

def fast_interpolate(v,inmesh,triidxs,interpweights):
    idxs = inmesh.cells[1].data[triidxs]
    return np.sum(v[idxs]*interpweights,axis=1)