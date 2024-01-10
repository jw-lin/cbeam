import numpy as np
from wavesolve.fe_solver import solve_waveguide,get_eff_index,construct_AB,solve_sparse,construct_B
from optics import load_meshio_mesh
from scipy.interpolate import UnivariateSpline,interp1d
import FEinterp
import copy,time,os
from scipy.integrate import solve_ivp

class prop:
    """ class for coupled mode propagation of tapered waveguides """
    def __init__(self,wl,wvg=None,Nmax=None,save_dir=None):
        """
        ARGS:
        wl: propagation wavelength
        wvg: waveguide to propagate through
        Nmax: the number of propagating modes throughout the waveguide
        save_dir: directory path to save output computations. default is './data/'
        """
        self.wvg = wvg
        self.wl = wl
        self.Nmax = Nmax
        self.k = 2*np.pi/wl

        self.cmat = None
        self.neff = None
        self.vs = None
        self.tapervals = None

        self.cmat_funcs = None
        self.neff_funcs = None
        self.v_func = None
        self.taper_func = None

        if save_dir is None:
            self.save_dir = './data'
        else:
            self.save_dir = save_dir

        self.check_and_make_folders()
    
    def get_prop_constants(self,z_arr,sparse=True):
        neffs = []
        self.wvg.update(z_arr[0])
        mesh = self.wvg.make_mesh()
        IOR_dict = self.wvg.assign_IOR()
        for i,z in enumerate(z_arr[1:]):
            scale_fac = self.wvg.taper_func(z)/self.wvg.taper_func(z_arr[i])
            mesh.points *= scale_fac
            w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=sparse,Nmax=self.Nmax)
            neffs.append(get_eff_index(self.wl,w))
        neffs = np.array(neffs)
        return neffs
    
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
    
    def make_sign_consistent(self,v,_v):
        """ alter the eigenmodes _v, assumed to be similar to v, so that they 
        have consistent overall sign with v. """

        flip_mask = np.sum(np.abs(v-_v),axis=1) > np.sum(np.abs(v+_v),axis=1)
        _v[flip_mask] *= -1
        return flip_mask
    
    def est_cross_term(self,B,v,_v,dz,vm=None):
        """ estimate the inner product between eigenmode derivatives and eigenmodes.
        ARGS:
        B: the B matrix ('mass matrix') in the generalized eigenvalue problem, used to solve for instantaneous eigenmodes
        v: an (N,M) array of N eigenmodes computed over M mesh points
        _v: the instanteous eigenmodes computed at a distance dz after where v was computed
        dz: the z distance separating the two sets of eigenmodes
        vm: the eigenmode basis halfway between v and _v. if None, this is approximated by averaging v and _v
        """
        dvdz = (_v-v)/dz # centered finite difference
        if vm is None:
            vavg = (v+_v)/2.
        else:
            vavg = vm
        return (B.dot(dvdz.T)).T.dot(vavg.T)

    def find_degenerate_groups(self,neffs,min_dif=1e-4):
        groups = []
        idxs = []
        for i in range(len(neffs)):
            group = []
            if i in idxs:
                continue
            for j in range(i+1,len(neffs)):
                if neffs[i] - neffs[j] < min_dif:
                    if not group:
                        group.append(i)
                        idxs.append(i)
                    group.append(j)
                    idxs.append(j)
            if group:
                groups.append(group)
        return groups

    def find_degenerate_groups_2(self,neffs,min_dif=1e-4):
        """ any eigenvalues w/in min_dif are treated as degenerate. a chain of eigenvalues
            with adjacent differences all less than min_dif are treated as a single 
            degenerate group.
        """
        groups = []
        idxs = []
        for i in range(len(neffs)):
            if i in idxs:
                continue
            group = [i]
            for j in range(i+1,len(neffs)):
                if neffs[group][-1] - neffs[j] < min_dif:
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

    def update_degen_groups(self,neffs,min_dif,last_degen_groups):
        """ this func assumes the degenerate groups can only become larger
            (so the eigenvalues of the waveguide are converging with propagation.)
            i believe this is the only way for the problem to be well defined.
        """
        degen_groups = copy.deepcopy(last_degen_groups)
        degen_groups_flat = [x for gr in degen_groups for x in gr]

        # look for adjacent neff values not already in a group and fit them together
        for i in range(1,len(neffs)):
            if i in degen_groups_flat or i-1 in degen_groups_flat:
                continue
            if neffs[i-1]-neffs[i] < min_dif:
                # find the correct index to insert at
                if len(degen_groups) == []:
                    degen_groups.append([i-1,i])
                else:
                    for j in range(len(degen_groups)):
                        if i < degen_groups[j][0]:
                            degen_groups.insert(j,[i-1,i])
                degen_groups_flat.append(i-1)
                degen_groups_flat.append(i)

        # look at neff values not already in a group and try to fit them into each degen group
        for i in range(len(neffs)):
            b = neffs[i]
            if i in degen_groups_flat:
                continue
            elif len(degen_groups) == 0:
                continue
            for gr in degen_groups:
                if i < gr[0] and b-neffs[gr[-1]] < min_dif:
                    gr.insert(0,i)
                    break
                elif i > gr[-1] and neffs[gr[0]] - b < min_dif:
                    gr.append(i)
                    break

        # look at groups and try to combine if the total difference is less than min_dif
        if len(degen_groups)>1:
            i = 1
            while i < len(degen_groups):
                if neffs[degen_groups[i-1][0]] - neffs[degen_groups[i][-1]] < min_dif:
                    biggroup = degen_groups[i-1]+degen_groups[i]
                    degen_groups[i] = biggroup
                    degen_groups.pop(i-1)
                else:
                    i+=1
        return degen_groups

    def avg_degen_neff(self,groups,neffs):
        for gr in groups:
            neffs[gr] = np.mean(neffs[gr]) 
        return neffs                            

    def compute(self,z,zi,mesh,_mesh,IOR_dict,points0,dz0,neffs,vlast=None,fixed_degen=[]):
        scale_facm = self.wvg.taper_func(z-dz0/2)/self.wvg.taper_func(zi)
        scale_facp = self.wvg.taper_func(z+dz0/2)/self.wvg.taper_func(zi)

        # minus step
        mesh.points = points0*scale_facm
        w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
        tripoints = mesh.points[mesh.cells[1].data,:2]
        meshtree = FEinterp.create_tree(tripoints[:,:3,:])

        # plus step
        _mesh.points = points0*scale_facp
        _w,_v,_N = solve_waveguide(_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
        _A,_B = construct_AB(_mesh,IOR_dict,self.k,sparse=True)

        # sign correction
        if vlast is None:
            self.make_sign_consistent(v,_v)
            # degeneracy correction
            for gr in fixed_degen:
                self.correct_degeneracy(gr,v,_v)
        else:
            self.make_sign_consistent(vlast,v)
            self.make_sign_consistent(vlast,_v)
            for gr in fixed_degen:
                self.correct_degeneracy(gr,vlast,v)
                self.correct_degeneracy(gr,vlast,_v)

        vlast = 0.5*(v+_v)

        # interpolation

        vinterp = np.copy(_v)

        triidxs,interpweights = FEinterp.get_idxs_and_weights(_mesh.points,meshtree)

        mask = (triidxs != -1)
        for k,vec in enumerate(v):
            vinterp[k,:][mask] = fast_interpolate(vec,mesh,triidxs[mask],interpweights[mask])

        # est deriv
        cmat = self.est_cross_term(_B,vinterp,_v,dz0)
        neffs = get_eff_index(self.wl,0.5*(w+_w))
        return cmat,neffs,vlast

    def compute_cmat_norm(self,cmat,fixed_degen=[]):
        """ compute a matrix 'norm' that will be used to check accuracy in adaptive z-step scheme. """
        ## when computing norm, ignore diagonal terms and any terms that are in a degenerate group, these are numerical noise
        # this could be sped up ...
        cnorm = 0
        for i in range(self.Nmax):
            for j in range(self.Nmax):
                if i==j:
                    continue
                for gr in fixed_degen:
                    if i in gr and j in gr:
                        continue
                cnorm += np.abs(cmat[i,j])
        return cnorm
    
    def prop_setup(self,zi,zf,max_interp_error=3e-4,dz0=1,min_zstep=1,save=False,tag="",fixed_degen=[],fixed_step=None):
        """ compute the eigenmodes, eigenvalues (effective indices), and cross-coupling coefficients 
            for the waveguide loaded into self.wvg, in the interval [zi,zf]. Uses an adaptive 
            scheme (based on interpolation of previous data points) to choose the z step.
        ARGS:
            zi: initial z coordinate for waveguide modelling (doesn't have to be 0)
            zf: final z coordinate
            max_interp_error: how carefully we step forwards in z. lower -> smaller steps
            dz0: the default delta-z used to numerically estimate eigenmode derivatives
            min_zstep: if the adaptive scheme chooses a z-step less than this value, an exception is raised
            save: set True to write outputs to file; they can be loaded with self.load()
            tag: the unique string to associate to a computation, used to load() it later
            fixed_degen: manually specify groups of degenerate modes (by index), improving convergence (hopefully)
            fixed_step: manually set a fixed z step, ignoring the adaptive stepping scheme

        RETURNS:
            zs: array of z values
        """
        start_time = time.time()
        zstep0 = 10 if fixed_step is None else fixed_step # starting step

        cmats = []
        cmats_norm = []
        neffs = []
        vs = []
        zs = []
        tapervals=[]

        self.wvg.update(0)

        #always init the mesh from 0
        ps = "_"+tag if tag is not None else ""
        if save:
            meshwriteto=self.save_dir+"/meshes/mesh"+ps
        else:
            meshwriteto=None
        print("generating mesh...")
        mesh = self.wvg.make_mesh_bndry_ref(size_scale_fac=1.,min_mesh_size=0.4,max_mesh_size=10.,_power=1,writeto=meshwriteto,_align=False)
        _mesh = copy.deepcopy(mesh)
        
        IOR_dict = self.wvg.assign_IOR()

        points0 = np.copy(mesh.points * self.wvg.taper_func(zi))
        
        vlast = None

        z = zi
        print("starting computation ...")
        while True:
            if zstep0<min_zstep:
                print("comp. halted")
                raise Exception("error: zstep has decreased below lower limit - eigenmodes may not be varying continuously")
            if fixed_step:
                dz = fixed_step
            else:
                dz = min(zstep0/10,dz0)
            cmat,neff,vlast_temp = self.compute(z,zi,mesh,_mesh,IOR_dict,points0,dz,neffs,vlast=vlast,fixed_degen=fixed_degen)

            cnorm = self.compute_cmat_norm(cmat,fixed_degen)

            if len(zs)<4 or fixed_step:
                cmats.append(cmat)
                neffs.append(neff)
                cmats_norm.append(cnorm)
                zs.append(z)
                tapervals.append(self.wvg.taper_func(z))
                vlast = vlast_temp
                vs.append(vlast)
                if z == zf:
                    break
                z = min(zf,z+zstep0)
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                continue

            # construct spline to test if current z step is small enough
            spl = UnivariateSpline(zs[-4:],cmats_norm[-4:],k=3,s=0,ext=0)
            err = np.abs(spl(z)-cnorm)
            if err<max_interp_error:
                cmats.append(cmat)
                neffs.append(neff)
                cmats_norm.append(cnorm)
                zs.append(z)
                tapervals.append(self.wvg.taper_func(z))
                vlast = vlast_temp
                vs.append(vlast)
                if z == zf:
                    break
                # rescale zstep0
                if err<0.1*max_interp_error:
                    zstep0*=2
                z = min(zf,z+zstep0) 
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
            else:
                print("\rcurrent z: {0} / {1}; tol. not met, reducing step        ".format(z,zf),end='',flush=True)             
                z -= zstep0
                zstep0/=2
                z += zstep0
        
        vs = np.array(vs).T # eigenmode array is (MxNxK) for M mesh points, N eigenmodes, and K z values
        
        if save:
            self.save(zs,tapervals,cmats,neffs,vs,tag=tag)
        print("time elapsed: ",time.time()-start_time)

        self.cmat = cmats
        self.neff = neffs
        self.vs = vs
        self.za = zs
        self.mesh = mesh
        self.tapervals = tapervals
        return zs,tapervals,cmats,neffs,vs,mesh

    def check_and_make_folders(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.save_dir+'/eigenmodes'):
            os.makedirs(self.save_dir+'/eigenmodes')
        if not os.path.exists(self.save_dir+'/eigenvalues'):
            os.makedirs(self.save_dir+'/eigenvalues')
        if not os.path.exists(self.save_dir+'/cplcoeffs'):
            os.makedirs(self.save_dir+'/cplcoeffs')
        if not os.path.exists(self.save_dir+'/zvals'):
            os.makedirs(self.save_dir+'/zvals')
        if not os.path.exists(self.save_dir+'/meshes'):
            os.makedirs(self.save_dir+'/meshes')
        if not os.path.exists(self.save_dir+'/tapervals'):
            os.makedirs(self.save_dir+'/tapervals')

    def save(self,zs,tapervals,cmats,neffs,vs,tag=""):
        ps = "" if tag == "" else "_"+tag
        np.save(self.save_dir+'/cplcoeffs/cplcoeffs'+ps,cmats)
        np.save(self.save_dir+'/eigenmodes/eigenmodes'+ps,vs)
        np.save(self.save_dir+'/eigenvalues/eigenvalues'+ps,neffs)
        np.save(self.save_dir+'/zvals/zvals'+ps,zs)
        np.save(self.save_dir+'/tapervals/tapervals'+ps,tapervals)

    def load(self,tag=""):
        ps = "" if tag == "" else "_"+tag
        self.cmat = np.load(self.save_dir+'/cplcoeffs/cplcoeffs'+ps+'.npy')
        self.neff = np.load(self.save_dir+'/eigenvalues/eigenvalues'+ps+'.npy')
        self.vs = np.load(self.save_dir+'/eigenmodes/eigenmodes'+ps+'.npy')
        self.za = np.load(self.save_dir+'/zvals/zvals'+ps+'.npy')
        try:
            self.tapervals = np.load(self.save_dir+'/tapervals/tapervals'+ps+".npy")
        except: 
            pass
        self.mesh = load_meshio_mesh(self.save_dir+'/meshes/mesh'+ps)
        if self.Nmax==None:
            self.Nmax = self.neff.shape[1]
    
    def make_interp_funcs(self):
        """ construct interpolation functions for coupling matrices and mode effective indices,
            loaded into self.cmat and self.neff, which were computed on an array of z values self.za.
        """
        za = self.za
        def make_c_func(i,j):
            assert i < j, "i must be < j in make_interp_funcs()"
            return UnivariateSpline(za,0.5*(self.cmat[:,i,j]-self.cmat[:,j,i]),ext=0,s=0)
        cmat_funcs = []
        for j in range(1,self.Nmax):
            for i in range(j):
                cmat_funcs.append(make_c_func(i,j))
        
        neff_funcs = []
        for i in range(self.Nmax):
            neff_funcs.append(UnivariateSpline(za,self.neff[:,i],s=0))
    
        self.cmat_funcs = cmat_funcs
        self.neff_funcs = neff_funcs
        self.neff_int_funcs = [neff_func.antiderivative() for neff_func in neff_funcs]
        try:
            self.v_func = interp1d(self.za,self.vs,assume_sorted=True)
        except:
            pass
        if self.tapervals is not None:
            self.taper_func = UnivariateSpline(self.za,self.tapervals,s=0)
    
    def compute_cmat(self,z):
        """ using interpolation, compute the cross-coupling matrix at z """
        out = np.zeros((self.Nmax,self.Nmax))
        k = 0
        for j in range(1,self.Nmax):
            for i in range(j):
                val = self.cmat_funcs[k](z)
                out[i,j] = -val
                out[j,i] = val
                k+=1
        return out
    
    def compute_neff(self,z):
        """ using interpolation, compute the array of mode effective indices at z """
        return np.array([neff(z) for neff in self.neff_funcs])
    
    def compute_int_neff(self,z):
        """ compute the antiderivative of the mode effective indices at z"""
        return np.array([neffi(z) for neffi in self.neff_int_funcs])

    def propagate(self,u0,zf):
        """ propagate a launch wavefront, expressed in the basis of initial eigenmodes, to z = zf 
        ARGS:
        u0: the launch field, expressed as mode amplitudes of the initial eigenmode basis
        zf: the z coordinate to propagate through to.

        RETURNS:
        za: the array of z values used by the ODE solver (RK23)
        u: the mode amplitudes of the wavefront, evaluated along za
        uf: the final mode amplitudes of the wavefront, accounting for overall phase evolution of the eigenmodes
        v: the final wavefront, computed over the finite element mesh
        """
        zi = self.za[0]
        def deriv(z,u):
            neffs = self.compute_neff(z)
            phases = self.k * (self.compute_int_neff(z)-self.compute_int_neff(zi))
            cmat = self.compute_cmat(z)
            phase_mat = np.exp(1.j * (phases[None,:] - phases[:,None]))
            return -1./neffs*np.dot(phase_mat*cmat,u*neffs)
        
        sol = solve_ivp(deriv,(zi,zf),u0,'RK23') # RK45 might be faster, but since RK23 tests more points, the cross-coupling behavior is more resolved
        # multiply by phase factors
        final_phase = np.exp(1.j*self.k*np.array(self.compute_int_neff(zf)-self.compute_int_neff(zi)))
        uf = sol.y[:,-1]*final_phase
        v = np.sum(uf[None,:]*self.vs[:,:,-1],axis=1)

        return sol.t,sol.y,uf,v

    def compute_change_of_basis(self,z,newbasis,u=None):
        """ compute the (Nmax x Nmax) change of basis matrix between the current eigenbasis at z and a new basis 
        ARGS: 
        z: z coordinate at which the currently loaded eigenbasis should be evaluated
        newbasis: MxN array of N eigenmodes computed over M mesh points, which we want to expand in
        u: (optional) Nx1 modal vector to express in new basis

        RETURNS:
        cob: Nmax x Nmax change of basis matrix
        _u: Nx1 modal vector corresponding to u expressed in the new basis. None, if u is not provided.
        """
        assert self.taper_func is not None,"run make_interp_funcs() first"
        _mesh = copy.deepcopy(self.mesh)
        _mesh.points *= self.taper_func(z)
        B = construct_B(_mesh,sparse=True)
        oldbasis = self.v_func(z)
        cob = B.dot(newbasis).T.dot(oldbasis)
        if u is not None:
            return cob,np.dot(cob,u)
        return cob,None

def fast_interpolate(v,inmesh,triidxs,interpweights):
    idxs = inmesh.cells[1].data[triidxs]
    return np.sum(v[idxs]*interpweights,axis=1)