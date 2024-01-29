import numpy as np
from wavesolve.fe_solver import solve_waveguide,get_eff_index,construct_AB,construct_B
from wavesolve.shape_funcs import compute_NN
from waveguide import load_meshio_mesh
from scipy.interpolate import UnivariateSpline,interp1d
import copy,time,os,FEval
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

def fast_interpolate(v,inmesh,triidxs,interpweights):
    idxs = inmesh.cells[1].data[triidxs]
    return np.sum(v[idxs]*interpweights,axis=1)

class Propagator:
    """ class for coupled mode propagation of tapered waveguides """

    ## high level parameters

    # propagation params
    solver = "RK45"         # solving method for scipy's solve_ivp
    WKB = False             # whether to add the WKB-like correction to the coupled mode equations (usually negligible)

    # z-stepping params
    zstep_tol = 1e-4        # controls how adaptive stepping level. lower = more steps (better z resolution)
    fixed_zstep = None      # set to a numeric value to use a fixed zstep
    coupling_dz = 0.1       # the z-step used to estimate derivative terms (which appear in calculation of the coupling coeffs.)
    min_zstep = 1.25
    max_zstep = 320.

    # misc params
    degen_crit = 0.         # the minimum difference in effective index two modes can have before they are considered degenerate    
    degen_groups = []       # used to explicitly specify degenerate mode groups in the guide

    mesh_mode = "transform" # how to update the mesh with z. either "transform" or "remesh". 
                            # in transform mode, the mesh is continuously transformed. the waveguide must support this.
                            # in remesh mode, a new mesh is generated at every z. this (in theory) allows for a wider
                            # range of waveguides, but in practice is less numerically stable. (work in progress)
    
    inner_product_mode = "integrate"    # how to compute the inner products. either "integrate" or "interpolate."
                                        # in integrate mode: inner products between fields computed on different meshes are numerically integrated using Grundmann-Moeller.
                                        # in interpolate mode: inner products are computed by evaulating one field on the the mesh of the other.
                                        # then, the inner product is estimated as v_1^T B v_2, where B is the "mass matrix" for the shared mesh.

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

        self.cmat_funcs = None
        self.neff_funcs = None
        self.compute_v = None
        self.points0 = None

        if save_dir is None:
            self.save_dir = './data'
        else:
            self.save_dir = save_dir

        self.check_and_make_folders()                    
    
    #region main functions

    def prop_setup(self,zi,zf,save=False,tag="",plot=False):
        if self.mesh_mode == "transform":
            return self.prop_setup_transform(zi,zf,save,tag,self.degen_groups,plot)
        elif self.mesh_mode == "remesh": 
            return self.prop_setup_remesh(zi,zf,save,tag,self.degen_groups,plot)

    def make_interp_funcs(self,za=None):
        """ construct interpolation functions for coupling matrices and mode effective indices,
            loaded into self.cmat and self.neff, which were computed on an array of z values self.za.
        """
        
        if za is None:
            za = np.copy(self.za)
        
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
        self.neff_dif_funcs = [neff_func.derivative() for neff_func in neff_funcs]

        if self.mesh_mode == "transform":
            self.compute_v = interp1d(za,self.vs,assume_sorted=True,axis=0)

    def propagate(self,u0,zi=None,zf=None):
        """ propagate a launch wavefront, expressed in the basis of initial eigenmodes, to z = zf 
        ARGS:
        u0: the launch field, expressed as mode amplitudes of the initial eigenmode basis
        zf: the z coordinate to propagate through to.

        RETURNS:
        za: the array of z values used by the ODE solver
        u: the mode amplitudes of the wavefront, evaluated along za
        uf: the final mode amplitudes of the wavefront, accounting for overall phase evolution of the eigenmodes
        v: the final wavefront, computed over the finite element mesh
        """
        u0 = np.array(u0,dtype=np.complex128)
        if zi is None:
            zi = self.za[0]
        if zf is None:
            zf = self.za[-1]

        def deriv(z,u):
            neffs = self.compute_neff(z)
            phases = (self.k * (self.compute_int_neff(z)-self.compute_int_neff(zi)))%(2*np.pi)
            cmat = self.compute_cmat(z)
            phase_mat = np.exp(1.j * (phases[None,:] - phases[:,None]))
            ddz = -1./neffs*np.dot(phase_mat*cmat,u*neffs)
            if self.WKB: 
                ddz += self.WKB_cor(z)*u
            return ddz
        
        sol = solve_ivp(deriv,(zi,zf),u0,self.solver,rtol=1e-12,atol=1e-10)
        # multiply by phase factors
        final_phase = np.exp(1.j*self.k*np.array(self.compute_int_neff(zf)-self.compute_int_neff(zi)))
        uf = sol.y[:,-1]*final_phase

        return sol.t,sol.y,uf

    #endregion

    #region setup computations
            
    def compute_transform(self,z,zi,mesh0,mesh,_mesh,IOR_dict,dz0,neffs,vlast=None,degen_groups=[],min_degen_dif=0):
        assert type(self.wvg).transform != self.wvg.transform, "waveguide does not have a transform() function that overides base!"
        self.wvg.transform_mesh(mesh0,zi,z-dz0/2,mesh)
        self.wvg.transform_mesh(mesh0,zi,z+dz0/2,_mesh)

        # minus step
        w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
        BVHtree = FEval.create_tree(mesh.points,mesh.cells[1].data) # note that updating the tree vs recreating doesn't seem to save any time

        # plus step
        _w,_v,_N = solve_waveguide(_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
        BVHtree2 = FEval.create_tree(_mesh.points,_mesh.cells[1].data)
        neffs = get_eff_index(self.wl,0.5*(w+_w))

        if min_degen_dif>0:
            degen_groups = self.update_degen_groups(get_eff_index(_w),min_degen_dif,degen_groups)

        # sign correction
        if vlast is None:
            self.make_sign_consistent(v,_v)
            # degeneracy correction
            for gr in degen_groups:
                self.correct_degeneracy(gr,v,_v)
                #self.avg_degen_neff(gr,neffs)
        else:
            self.make_sign_consistent(vlast,v)
            self.make_sign_consistent(vlast,_v)
            # degeneracy correction
            for gr in degen_groups:
                self.correct_degeneracy(gr,vlast,v)
                self.correct_degeneracy(gr,vlast,_v)
                #self.avg_degen_neff(gr,neffs)  
        vlast = 0.5*(v+_v)

        if self.inner_product_mode == "interpolate":
            vinterp = np.copy(_v)
            triidxs,interpweights = FEval.get_idxs_and_weights(_mesh.points,BVHtree)
            mask = (triidxs != -1)
            for k,vec in enumerate(v):
                vinterp[k,:][mask] = fast_interpolate(vec,mesh,triidxs[mask],interpweights[mask])

            # est deriv
            _A,_B = construct_AB(_mesh,IOR_dict,self.k,sparse=True)
            cmat = self.est_cross_term(_B,vinterp,_v,dz0)
        elif self.inner_product_mode == "integrate":
            cmat = -np.array(FEval.compute_coupling_simplex(v.T,BVHtree,_v.T,BVHtree2))/dz0
        else:
            raise Exception("inner_product_mode "+self.inner_product_mode+" not recognized")

        return cmat,neffs,vlast,degen_groups

    def compute_cmat_norm(self,cmat,degen_groups=[]):
        """ compute a matrix 'norm' that will be used to check accuracy in adaptive z-step scheme. """
        ## when computing norm, ignore diagonal terms and any terms that are in a degenerate group, these are numerical noise
        # this could be sped up ...
        cnorm = 0
        for i in range(self.Nmax):
            for j in range(self.Nmax):
                if i==j:
                    continue
                for gr in degen_groups:
                    if i in gr and j in gr:
                        continue
                cnorm += np.abs(cmat[i,j])
        return cnorm

    def prop_setup_transform(self,zi,zf,save=False,tag="",degen_groups=[],plot=False):
        """ compute the eigenmodes, eigenvalues (effective indices), and cross-coupling coefficients 
            for the waveguide loaded into self.wvg, in the interval [zi,zf]. Uses an adaptive 
            scheme (based on interpolation of previous data points) to choose the z step.
        """

        tol = self.zstep_tol
        fixed_step = self.fixed_zstep
        min_degen_dif = self.degen_crit
        min_zstep = self.min_zstep
        max_zstep = self.max_zstep

        dz0 = self.coupling_dz

        start_time = time.time()
        zstep0 = 10 if fixed_step is None else fixed_step # starting step

        self.wvg.update(zi)

        cmats = []
        cmats_norm = []
        neffs = []
        vs = []
        zs = []

        ps = "_"+tag if tag is not None else ""
        if save:
            meshwriteto=self.save_dir+"/meshes/mesh"+ps
        else:
            meshwriteto=None

        print("generating mesh...")
        mesh0 = self.generate_mesh(writeto=meshwriteto)
        mesh = copy.deepcopy(mesh0)
        _mesh = copy.deepcopy(mesh0)
        
        print("number of mesh points: ",mesh.points.shape[0])
        
        IOR_dict = self.wvg.assign_IOR()

        if plot:
            print("initial mesh: ")
            self.wvg.plot_mesh(mesh)
            print("initial modes: ")
            w,v,n = solve_waveguide(mesh0,self.wl,IOR_dict,plot=True,sparse=True,Nmax=self.Nmax)

        vlast = None

        z = zi
        print("starting computation ...")
        while True:
            if zstep0<min_zstep:
                zstep0 = min_zstep

            if fixed_step:
                dz = min(fixed_step/10,dz0)
            else:
                dz = min(zstep0/10,dz0)

            cmat,neff,vlast_temp,degen_groups = self.compute_transform(z,zi,mesh0,mesh,_mesh,IOR_dict,dz,neffs,vlast=vlast,degen_groups=degen_groups,min_degen_dif=min_degen_dif)

            cnorm = self.compute_cmat_norm(cmat,degen_groups)

            if len(zs)<4 or fixed_step or zstep0 == min_zstep: # always accept the computation under these conditions
                if zstep0 == min_zstep and len(zs)>4 and not fixed_step:
                    spl = UnivariateSpline(zs[-4:],cmats_norm[-4:],k=3,s=0,ext=0)
                    err = np.abs(spl(z)-cnorm)
                    if err<0.1*tol:
                        zstep0 = min(zstep0*2,max_zstep,zf-z-zstep0)

                cmats.append(cmat)
                neffs.append(neff)
                cmats_norm.append(cnorm)
                zs.append(z)
                vlast = vlast_temp
                vs.append(vlast)
                if z == zf:
                    break

                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                #print("accepted_cmat: ",cmat[1,0])
                z += zstep0
                continue

            spl = UnivariateSpline(zs[-4:],cmats_norm[-4:],k=3,s=0,ext=0)
            err = np.abs(spl(z)-cnorm)
            if err<tol: # accept if extrapolation error is sufficiently low
                cmats.append(cmat)
                neffs.append(neff)
                cmats_norm.append(cnorm)
                zs.append(z)
                vlast = vlast_temp
                vs.append(vlast)
                if z == zf:
                    break
                # rescale zstep0
                if err<0.1*tol:
                    zstep0 = min(zstep0*2,max_zstep, zf-z)
                else:
                    zstep0 = min(zstep0,max_zstep, zf-z)
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                #print("accepted_cmat: ",cmat[1,0])
                z += zstep0
            else:
                print("\rcurrent z: {0} / {1}; reducing step : {2}        ".format(z,zf,zstep0),end='',flush=True)             
                z -= zstep0
                zstep0 = min(max(zstep0/2,min_zstep),zf-z,max_zstep)
                z += zstep0
        
        if save:
            self.save(zs,cmats,neffs,vs,tag=tag)
        print("time elapsed: ",time.time()-start_time)

        self.cmat = np.array(cmats)
        self.neff = np.array(neffs)
        self.vs = np.array(vs)
        self.za = np.array(zs)
        self.mesh = mesh
        return self.za,self.cmat,self.neff,self.vs

    def compute_cmat_pert_on_isect_mesh(self,dz,w,v,mesh,isect_dict,degen_groups=[]):

        N = len(mesh.points)
        B_IORsq_diff = lil_matrix((N,N))
        materials = mesh.cell_sets.keys()
        for material in materials:
            tris = mesh.cells[1].data[tuple(mesh.cell_sets[material])][0,:,0,:]
            for tri in tris:
                ix = np.ix_(tri,tri)
                tri_points = mesh.points[tri]
                NN = compute_NN(tri_points)
                B_IORsq_diff[ix] += isect_dict[material] * NN

        mat = self.k**2*B_IORsq_diff.dot(v.T).T.dot(v.T)/dz
        wdif = w[:,None] - w[None,:]
        
        for i in range(self.Nmax):
            for j in range(i+1):
                if i == j:
                    mat[i,i] = 0.
                    wdif[i,i] = 1.
                    continue

                for gr in degen_groups:
                    if i in gr and j in gr:
                        mat[i,j] = 0
                        wdif[i,j] = 1.
                        mat[j,i] = 0
                        wdif[j,i] = 1.
                        continue
        return mat/wdif

    def compute_remesh(self,z,zi,mesh0,mesh,IOR_dict,cmats,vs,IORdif_dict,zstep0,vlast=None,degen_groups=[],min_degen_dif=0,mode="transform"):
        
        dz0 = zstep0 / 20

        if mode == "transform":
            new_mesh = self.wvg.transform_mesh(mesh0,zi,z)
        else:
            new_mesh,IOR_dict = self.generate_isect_mesh(z,dz0) 

        # compute eigenmodes
        w,v,N = solve_waveguide(new_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
        
        # look at degeneracy
        if min_degen_dif != 0:
            new_degen_groups = self.update_degen_groups(get_eff_index(self.wl,w),min_degen_dif,degen_groups)
            print("degenerate groups: ",new_degen_groups)
        else:
            new_degen_groups = degen_groups
        
        if mode == "transform":
            if vlast is not None:
                self.make_sign_consistent(vlast,v)
        else:
            # interpolation for now
            if vlast is not None:
                tree = FEval.create_tree(mesh.points,mesh.cells[1].data)
                vinterp = np.copy(v)
                triidxs,interpweights = FEval.get_idxs_and_weights(new_mesh.points,tree)
                mask = (triidxs != -1)
                for k,vec in enumerate(vlast):
                    vinterp[k,:][mask] = fast_interpolate(vec,mesh,triidxs[mask],interpweights[mask])
                self.make_sign_consistent(vinterp,v)

        # this works if the same modes are degenerate, or if modes have been added to the degenerate group.
        if len(degen_groups) == len(new_degen_groups):
            for gr in new_degen_groups:
                if vlast is not None:
                    if mode=="transform":
                        self.correct_degeneracy(gr,vlast,v)
                    else:
                        self.correct_degeneracy(gr,vinterp,v)
            new_cmats = cmats
            new_vs = vs
        
        # degeneracy splitting - need to add in a check to raise error on double split, at some point
        else:
            print("potential degeneracy splitting detected ... ")
            new_cmats = copy.deepcopy(cmats)
            new_vs = copy.deepcopy(vs)
            for gr in degen_groups:
                if mode == "transform":
                    v,vlast,Q = self.correct_degeneracy(gr,v,vlast) # we need to retroactively apply this change of basis 
                else:
                    v,vlast,Q = self.correct_degeneracy(gr,v,vinterp)
                for i,c in enumerate(cmats):
                    new_c =  np.dot(Q.T,np.dot(c[gr,gr],Q)) 
                    new_cmats[i][gr,gr] = new_c
                    new_v = np.dot(Q.T,vs[i][gr])
                    new_vs[i][gr] = new_v

        cmat = self.compute_cmat_pert_on_isect_mesh(dz0,w,v,new_mesh,IORdif_dict,new_degen_groups)

        return cmat,new_cmats,get_eff_index(self.wl,w),np.copy(v),new_vs,new_degen_groups,new_mesh

    def prop_setup_remesh(self,zi,zf,save=False,tag="",degen_groups=[],plot=False):
        """ compute coupling coeffs, remeshing at every z """

        zstep0 = 10
        tol = self.zstep_tol
        min_zstep = self.min_zstep
        max_zstep = self.max_zstep
        min_degen_dif = self.degen_crit

        start_time = time.time()
        

        ps = "_"+tag if tag is not None else ""
        if save:
            meshwriteto=self.save_dir+"/meshes/mesh"+ps
        else:
            meshwriteto=None

        print("generating mesh...")
        mesh0,IOR_dict = self.generate_isect_mesh(zi,zstep0/10,writeto=meshwriteto)
        IORdif_dict = self.wvg.IORsq_diff()
        

        self.wvg.update(zi)

        cmats = []
        cmats_norm = []
        neffs = []
        vs = []
        zs = []

        mesh = copy.deepcopy(mesh0)
        
        print("number of mesh points: ",mesh0.points.shape[0])

        vlast = None

        if plot:
            w,v,N = solve_waveguide(mesh0,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax,plot=True)

        z = zi
        print("starting computation ...")
        while True:
            if zstep0<min_zstep:
                zstep0 = min_zstep

            cmat,cmats_temp,neff,vlast_temp,vs_temp,degen_groups_temp,mesh_temp = self.compute_remesh(z,zi,mesh0,mesh,IOR_dict,cmats,vs,IORdif_dict,zstep0,vlast=vlast,degen_groups=degen_groups,min_degen_dif=min_degen_dif)
            
            cnorm = self.compute_cmat_norm(cmat,degen_groups)

            if len(zs)<4 or zstep0 == min_zstep: # always accept the computation under these conditions
                if zstep0 == min_zstep and len(zs)>4:
                    spl = UnivariateSpline(zs[-4:],cmats_norm[-4:],k=3,s=0,ext=0)
                    err = np.abs(spl(z)-cnorm)
                    if err<0.1*tol:
                        zstep0 = min(zstep0*2,max_zstep,zf-z-zstep0)

                cmats = cmats_temp
                vs = vs_temp
                mesh = mesh_temp
                cmats.append(cmat)
                neffs.append(neff)
                cmats_norm.append(cnorm)
                zs.append(z)
                vlast = vlast_temp
                degen_groups = degen_groups_temp
                
                vs.append(vlast)
                if z == zf:
                    break

                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                #print("accepted_cmat: ",cmat[0,1])
                z += zstep0
                continue

            spl = UnivariateSpline(zs[-4:],cmats_norm[-4:],k=3,s=0,ext=0)
            err = np.abs(spl(z)-cnorm)
            if err<tol: # accept if extrapolation error is sufficiently low
                cmats = cmats_temp
                vs = vs_temp
                mesh = mesh_temp
                cmats.append(cmat)
                neffs.append(neff)
                cmats_norm.append(cnorm)
                zs.append(z)
                vlast = vlast_temp
                degen_groups = degen_groups_temp
                vs.append(vlast)
                if z == zf:
                    break
                # rescale zstep0
                if err<0.1*tol:
                    zstep0 = min(zstep0*2,max_zstep, zf-z)
                else:
                    zstep0 = min(zstep0,max_zstep, zf-z)
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                #print("accepted_cmat: ",cmat[0,1])
                z += zstep0
            else:
                print("\rcurrent z: {0} / {1}; reducing step : {2}        ".format(z,zf,zstep0),end='',flush=True)             
                z -= zstep0
                zstep0 = min(max(zstep0/2,min_zstep),zf-z,max_zstep)
                z += zstep0
        
        if save:
            self.save(zs,cmats,neffs,vs,tag=tag)
        print("time elapsed: ",time.time()-start_time)

        self.cmat = np.array(cmats)
        self.neff = np.array(neffs)
        self.vs = vs
        self.za = np.array(zs)
        self.mesh = mesh
        return self.za,self.cmat,self.neff,self.vs

    def get_neffs(self,zi,zf=None,tol=1e-5,mesh=None):
        """ compute the effective refractive indices through a waveguide, using an adaptive step scheme. also saves interpolation functions to self.neff_funcs
        ARGS:
            zi: initial z coordinate
            zf: final z coordinate. if None, just return the value at zi
            tol: maximum error between an neff value computed at a proposed z step and the extrapolated value from previous points
            mesh: (optional) a mesh object if you don't want to use the auto-generated, default mesh
        RETURNS:
            zs: array of z coordinates
            neffs: array of effective indices computed on zs
        """
        start_time = time.time()
        neffs = []
        zs = []
        self.wvg.update(0)
        if mesh is None:
            mesh = self.generate_mesh() # use default vals
        points0 = np.copy(mesh.points)
        zstep0 = 10

        IOR_dict = self.wvg.assign_IOR()
        z = zi
        
        print("computing effective indices ...")

        if zf is None:
            w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            return z , get_eff_index(self.wl,w)

        while True:
            scale_fac = self.wvg.taper_func(z)/self.wvg.taper_func(zi)
            mesh.points = scale_fac*points0
            w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            neff = get_eff_index(self.wl,w)
            if len(neffs)<4:
                neffs.append(neff)
                zs.append(z)
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                if z == zf:
                    break
                z = min(zf,z+zstep0)
                continue
            
            # interpolate
            neff_interp = interp1d(zs[-4:],np.array(neffs)[-4:,:],kind='cubic',axis=0,fill_value="extrapolate")
            err = np.sum(np.abs(neff-neff_interp(z)))
            if err < tol:
                neffs.append(neff)
                zs.append(z)
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                if z == zf:
                    break
                if err < 0.1 * tol:
                    zstep0*=2
                z = min(zf,z+zstep0)
            else:
                print("\rcurrent z: {0} / {1}; tol. not met, reducing step        ".format(z,zf),end='',flush=True)   
                z -= zstep0 
                zstep0/=2
                z += zstep0
        
        neffs = np.array(neffs)
        neff_funcs = []
        zs = np.array(zs)
        for i in range(self.Nmax):
            neff_funcs.append(UnivariateSpline(zs,neffs[:,i],s=0))
        self.neff_funcs = neff_funcs
        self.neff = neffs
        print("time elapsed: ",time.time()-start_time)
        return zs , neffs
    
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

    def correct_degeneracy(self,group,v,_v,q=None):
        """ used least-squares minimization to transform modes _v so that they match v. mutates _v
        ARGS:
            group: a list of mode indexes setting which modes are to be considered degenerate.
            v: the mode basis we want to match
            _v: the mode basis we want to transform. We are allowed to apply a unitary transformation over the 
                modes specified by group.
            q (opt.): if not None, q is taken as the transformation matrix. if None, q is computed through lsq min.
        RETURNS:
            v: the original mode basis
            _v: the transformed mode basis
            q: the change-of-basis matrix
        """
        if q is None:
            coeff_mat = np.dot(v[group,:],_v[group,:].T)
            u,s,vh = np.linalg.svd(coeff_mat)
            q = np.dot(vh.T,u.T)
        _vq = np.dot(_v[group,:].T,q)
        for i,g in enumerate(group):
            _v[g] = _vq[:,i]
        return v,_v,q

    @staticmethod
    def update_degen_groups(neffs,dif,last_degen_groups,allow_split=True):
        """ Update an array, degen_groups which groups the indices of degenerate modes.
        ARGS:
            neffs: an array of mode effective indices
            dif: modes whose neff values differ by less than this are considered degenerate
            last_degen_groups: the previous value of degen_groups
            allow_split (bool): if True, the function is allowed to split degenerate groups of modes.
                                if False, degenerate mode groups can never shrink.
        RETURNS:
            degen_groups: a nested list which groups degenerate modes. e.g. [[0,1],[2,3]] means
                          that modes 0 and 1 are degenerate, and modes 2 and 3 are degenerate.
        """
        if dif==0:
            return last_degen_groups
        
        degen_groups = copy.deepcopy(last_degen_groups)
        degen_groups_flat = [x for gr in degen_groups for x in gr]

        # look for adjacent neff values not already in a group and fit them together
        for i in range(1,len(neffs)):
            if i in degen_groups_flat or i-1 in degen_groups_flat:
                continue
            if neffs[i-1]-neffs[i] < dif:
                # find the correct index to insert at
                if len(degen_groups) == 0:
                    degen_groups.append([i-1,i])
                else:
                    for j in range(len(degen_groups)):
                        if i < degen_groups[j][0]:
                            degen_groups.insert(j,[i-1,i])
                    degen_groups.insert(j+1,[i-1,i])
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
                if i < gr[0] and b-neffs[gr[-1]] < dif:
                    gr.insert(0,i)
                    break
                elif i > gr[-1] and neffs[gr[0]] - b < dif:
                    gr.append(i)
                    break

        # look at groups and try to combine if the total difference is less than min_dif
        if len(degen_groups)>1:
            i = 1
            while i < len(degen_groups):
                if neffs[degen_groups[i-1][0]] - neffs[degen_groups[i][-1]] < dif:
                    biggroup = degen_groups[i-1]+degen_groups[i]
                    degen_groups[i] = biggroup
                    degen_groups.pop(i-1)
                else:
                    i+=1
        
        if not allow_split:
            return degen_groups
        
        # look at groups and split if any adjacent difference is larger than min_dif
        for k,gr in enumerate(degen_groups):
            if len(gr)>1:
                i = 1
                while i < len(gr):
                    if neffs[gr[i-1]] - neffs[gr[i]] > dif:
                        gr1 = gr[0:i]
                        gr2 = gr[i:]
                        degen_groups[k] = gr1
                        degen_groups.insert(k+1,gr2)
                    i += 1
        return degen_groups

    def avg_degen_neff(self,group,neffs):
        neffs[group] = np.mean(neffs[group])[None] 
        return neffs

    #endregion

    #region utility

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

    def save(self,zs,cmats,neffs,vs,tag=""):
        ps = "" if tag == "" else "_"+tag
        if self.mesh_mode == "transform":
            vs = np.array(vs) # eigenmode array is (KxNxM) for M mesh points, N eigenmodes, and K z values
            np.save(self.save_dir+'/eigenmodes/eigenmodes'+ps,vs)
        else:
            # save the starting and ending eigenmodes
            np.save(self.save_dir+'/eigenmodes/eigenmodes_i'+ps,vs[0])
            np.save(self.save_dir+'/eigenmodes/eigenmodes_f'+ps,vs[-1])

        np.save(self.save_dir+'/cplcoeffs/cplcoeffs'+ps,cmats)
        np.save(self.save_dir+'/eigenvalues/eigenvalues'+ps,neffs)
        np.save(self.save_dir+'/zvals/zvals'+ps,zs)

    def load(self,tag=""):
        ps = "" if tag == "" else "_"+tag
        self.cmat = np.load(self.save_dir+'/cplcoeffs/cplcoeffs'+ps+'.npy')
        self.neff = np.load(self.save_dir+'/eigenvalues/eigenvalues'+ps+'.npy')
        if self.mesh_mode == "transform":
            self.vs = np.load(self.save_dir+'/eigenmodes/eigenmodes'+ps+'.npy')
        else:
            self.vs = [np.load(self.save_dir+'/eigenmodes/eigenmodes_i'+ps+".npy"),np.load(self.save_dir+'/eigenmodes/eigenmodes_f'+ps+".npy")]
        self.za = np.load(self.save_dir+'/zvals/zvals'+ps+'.npy')

        self.mesh = load_meshio_mesh(self.save_dir+'/meshes/mesh'+ps)
        self.points0 = np.copy(self.mesh.points)
        if self.Nmax==None:
            self.Nmax = self.neff.shape[1]
    
    #endregion

    #region aux funcs for interpolation/propagation
            
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

    def compute_dif_neff(self,z):
        return np.array([neffd (z) for neffd in self.neff_dif_funcs]) 

    def WKB_cor(self,z):
        dbeta_dz = self.k * self.compute_dif_neff(z) 
        return -0.5 * dbeta_dz / self.compute_neff(z)

    def compute_change_of_basis(self,newbasis,z,m,u=None):
        """ compute the (N x N) change of basis matrix between the current N-dimensional eigenbasis at z and a new basis 
        ARGS: 
        newbasis: MxN array of N eigenmodes computed over M mesh points, which we want to expand in
        z: z coordinate at which the currently loaded eigenbasis should be evaluated
        u: (optional) Nx1 modal vector to express in new basis

        RETURNS:
        cob: Nmax x Nmax change of basis matrix
        _u: Nx1 modal vector corresponding to u expressed in the new basis. None, if u is not provided.
        """

        B = construct_B(m,sparse=True)
        oldbasis = self.compute_v(z)
        cob = B.dot(newbasis.T).T.dot(oldbasis.T)

        if u is not None:
            return cob,np.dot(cob,u)
        return cob,None
    
    #endregion
        
    #region mesh gen

    def generate_mesh(self,writeto=None):
        """ generate a mesh for the loaded waveguide according to class attributes.
        ARGS:
            writeto (str or None): set to a str (the filename) to write the mesh to file. 
        """
        return self.wvg.make_mesh_bndry_ref(writeto=writeto)

    def generate_isect_mesh(self,z,dz,writeto=None):
        """ generate a mesh for the loaded waveguide formed by the intersection of waveguide boundaries (material interfaces) at 
            z and z+dz. this is used to compute coupling coefficients through perturbation theory (used when self.mesh_mode="transform").
        ARGS:
            z: the z coordinate that sets the first set of boundaries
            dz: the change in z that sets the second set of boundaries
            writeto (str or None): set to a str (the filename) to write the mesh to file. 
        """
        m,d = self.wvg.make_intersection_mesh(z,dz,writeto=writeto)
        return m,d    

    #endregion