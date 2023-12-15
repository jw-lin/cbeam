import numpy as np
from wavesolve.fe_solver import solve_waveguide,get_eff_index,construct_B,plot_eigenvector
from cbeam.waveguide import load_meshio_mesh,Waveguide,plot_mesh
from scipy.interpolate import UnivariateSpline,interp1d,CubicSpline
import copy,time,os
from cbeam import FEval
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from matplotlib import animation
from typing import Union

normcmap = np.zeros([256, 4])
normcmap[:, 3] = np.linspace(0, 1, 256)[::-1]
normcmap = ListedColormap(normcmap)

def plot_cfield(field,mesh,fig=None,ax=None,show_mesh=False,res=120,tree=None):
    show = False
    if ax is None:
        fig,ax = plt.subplots(1,1)
        show = True

    xlim = np.max(mesh.points[:,0])
    ylim = np.max(mesh.points[:,1])
    xa = np.linspace(-xlim,xlim,res)
    ya = np.linspace(-ylim,ylim,res)
    tree = FEval.create_tree_from_mesh(mesh) if tree is None else tree
    fgrid = np.array(FEval.evaluate_grid(xa,ya,field,tree)).T

    im = ax.imshow(np.arctan2(np.imag(fgrid),np.real(fgrid)),extent=(-xlim,xlim,-ylim,ylim),cmap="hsv",vmin=-np.pi,vmax=np.pi)
    ax.imshow(np.abs(fgrid),extent=(-xlim,xlim,-ylim,ylim),cmap=normcmap,interpolation="bicubic")

    if show_mesh:
        plot_mesh(mesh,plot_points=False,ax=ax,alpha=0.15)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    if fig is not None:
        fig.colorbar(im,ax=ax,label="phase")
    if show:
        plt.show()
    return im

def plot_field(field,mesh,ax=None,show_mesh=False):
    """ plot a real-valued finite element field, evaluated on the points of the mesh corresponding to the waveguide cross-section at z
    
    ARGS:
        field: the finite element field to plot, e.g. the output of make_field()
        mesh (opt.): the mesh object defining the points that field is evaluated on. If None, a mesh will be auto-generated
        ax (opt.): a matplotlib axis where the plot should go. if not None, you will need to call matplotlib.pyplot.show() manually
        show_mesh (opt.): whether or not to draw the mesh in the plot
    """
    show = False
    if ax is None:
        fig,ax = plt.subplots(1,1)
        show = True
    plot_eigenvector(mesh,field,show_mesh,ax,show)

class Propagator:
    """ class for coupled mode propagation of tapered waveguides """

    ## high level parameters

    # propagation params
    
    #: str: solving method for scipy's solve_ivp, default "RK45"
    solver = "RK45"
    #: bool: whether to add the WKB-like correction to the coupled mode equations (usually negligible)
    WKB = False

    # z-stepping params

    #: float: controls how adaptive stepping level. lower = more steps (better z resolution)
    zstep_tol = 1e-3
    #: float or None: set to a numeric value to use a fixed zstep
    fixed_zstep = None     
    #: float: minimum z step value when computing modes 
    min_zstep = 1.25
    #: float: maximum z step value when computing modes
    max_zstep = 640.

    # misc params
    degen_crit = 0.         # the minimum difference in effective index two modes can have before they are considered degenerate    
    
    #: list[list]: used to explicitly specify degenerate mode groups in the guide; this can greatly speed up computation.
    degen_groups = []

    #: str: mode for coupling matrix calculation, "from_wvg" or "from_interp"
    cmat_correction_mode = "from_interp"

    def __init__(self,wl,wvg:Union[None,Waveguide]=None,Nmax=None,save_dir=None):
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

        self.cmats = None
        self.neffs = None
        self.vs = None
        self.mesh = None
        self.zs = None

        self.cmats_funcs = None
        self.neffs_funcs = None
        self.get_v = None
        self.points0 = None

        self.channel_basis_matrix = None

        if save_dir is None:
            self.save_dir = './data'
        else:
            self.save_dir = save_dir

        self.check_and_make_folders()                    
    
    #region main functions

    def solve_at(self,z=0):
        """ solve for waveguide modes at given z value. returns effective indices and eigenmodes.
        
        ARGS:
            z (float) : z coordinate along the waveguide for the solve
        
        RETURNS:
            (tuple) : a tuple containing:

                - neff : an array of effective indices
                - v : an array of eigenmodes, dimensions NxM for N modes and M mesh points
        """

        mesh = self.make_mesh_at_z(z)
        IOR_dict = self.wvg.assign_IOR()
        w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
        self.mesh=mesh
        return get_eff_index(self.wl,w),v

    def characterize(self,zi=None,zf=None,mesh=None,tag='',save=False):
        """ compute the modes and coupling coefficients of the currently loaded waveguide
        
        ARGS:
            zi: initial z coordinate, default 0
            zf: final z coordinate. if None, just return the value at zi
            mesh: (optional) a mesh object if you don't want to use the auto-generated, default mesh
            tag: a string identifier which will be attached to any saved files
            save: set True to save function output (z values, effective indices, modes)
        
        RETURNS:
            (tuple) : a tuple containing:

                - zs: array of z coordinates
                - neffs: array of effective indices computed on zs; 0th axis is "z axis"
                - vs: eigenmodes computed on zs; 0th axis is "z axis"
                - cmats: coupling coefficients computed on zs; 0th axis is "z axis"
        """
        
        start_time = time.time()
        self.compute_modes(zi,zf,mesh,tag,save)
        self.compute_cmats(save=save,tag=tag)
        print("time elapsed: ",time.time()-start_time)
        self.make_interp_funcs()
        return self.zs , self.neffs , self.vs , self.cmats

    # alias
    prop_setup = characterize

    def propagate(self,u0,zi=None,zf=None):
        """ propagate a launch wavefront, expressed in the basis of initial eigenmodes, to z = zf 
        
        ARGS:
            u0: the launch field, expressed as mode amplitudes of the initial eigenmode basis.
            zi: the initial z coordinate corresponding to u0. if None, use the initial z value used in characterize().
            zf: the final z coordinate to propagate through to. if None, use the final z value used in characterize().

        RETURNS:
            (tuple): a tuple containing:
                - zs: the array of z values used by the ODE solver
                - u: the mode amplitudes of the wavefront, evaluated along za, with :math:`e^{i\\beta_j z}` phase factored *out*.
                - uf: the final mode amplitudes of the wavefront, with :math:`e^{i\\beta_j z}` phase factored *in*.
        """
  
        if zi is None:
            zi = self.zs[0]
        if zf is None:
            zf = self.zs[-1]

        if zi > zf:
            return self.backpropagate(u0,zi,zf)

        u0 = np.array(u0,dtype=np.complex128)
    
        def deriv(z,u):
            neffs = self.get_neff(z)
            phases = (self.k * (self.get_int_neff(z)-self.get_int_neff(zi)))%(2*np.pi)
            cmat = self.get_cmat(z)
            phase_mat = np.exp(1.j * (phases[None,:] - phases[:,None]))
            ddz = -1./neffs*np.dot(phase_mat*cmat,u*neffs)
            if self.WKB: 
                ddz += self.WKB_cor(z)*u
            return ddz
        
        sol = solve_ivp(deriv,(zi,zf),u0,self.solver,rtol=1e-12,atol=1e-10)
        # multiply by phase factors
        uf = self.apply_phase(sol.y[:,-1],sol.t[-1])
        return sol.t,sol.y,uf

    def apply_phase(self,u,z,zi=None):
        """ apply :math:`e^{i \\beta_j z}` phase variation to the mode amplitude of eigenmode j
        with propagation constant :math:`\\beta_j`.
        
        ARGS:
            u: mode amplitude vector
            z: z coord. that u is evaluated at
            zi (opt.) the starting z coordinate to measure the change in phase from. 
        """
        if zi is None:
            zi = self.zs[0]
        phase = np.exp(1.j*self.k*np.array(self.get_int_neff(z)-self.get_int_neff(zi)))
        return u*phase
    
    def backpropagate(self,u0,zf=None,zi=None):
        """ propagate a wavefront from the back of the waveguide to the front """
        
        u0 = np.array(u0,dtype=np.complex128)
        if zi is None:
            zi = self.zs[0]
        if zf is None:
            zf = self.zs[-1]

        def deriv(z,u):
            zp = self.zs[-1] - z
            neffs = self.get_neff(zp)
            phases = (self.k * (self.get_int_neff(zp)-self.get_int_neff(zf)))%(2*np.pi)
            cmat = self.get_cmat(zp)
            phase_mat = np.exp(1.j * (phases[None,:] - phases[:,None]))
            ddz = -1./neffs*np.dot(phase_mat*cmat,u*neffs)
            if self.WKB: 
                ddz += self.WKB_cor(zp)*u
            return -ddz

        sol = solve_ivp(deriv,(self.zs[-1]-zf,self.zs[-1]-zi),u0,self.solver,rtol=1e-12,atol=1e-10)
        # multiply by phase factors
        uf = self.apply_phase(sol.y[:,-1],zi,zf)
        return sol.t,sol.y,uf

    #endregion

    #region other setup funcs
    def compute_neffs(self,zi,zf=None,tol=1e-5,mesh=None):
        """ compute the effective refractive indices through a waveguide, using an adaptive step scheme. also saves interpolation functions to self.neffs_funcs
        
        ARGS:
            zi: initial z coordinate
            zf: final z coordinate. if None, just return the value at zi
            tol: maximum error between an neff value computed at a proposed z step and the extrapolated value from previous points
            mesh: (optional) a mesh object if you don't want to use the auto-generated, default mesh
        
        RETURNS:
            (tuple): a tuple containing:
                - zs: array of z coordinates
                - neffs: array of effective indices computed on zs
        """
        start_time = time.time()
        neffs = []
        zs = []
        vs = []
        self.wvg.update(0)
        if mesh is None:
            mesh = self.generate_mesh() # use default vals
        self.mesh = mesh
        zstep0 = 10
        min_zstep = 10

        IOR_dict = self.wvg.assign_IOR()
        z = zi
        
        print("computing effective indices ...")

        if zf is None:
            w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            return z , get_eff_index(self.wl,w) , v

        vlast = None

        while True:
            _mesh = self.wvg.transform_mesh(mesh,0,z)
            w,v,N = solve_waveguide(_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            if vlast is not None:
                v,w = self.swap_modes(vlast,v,w)
                self.make_sign_consistent(vlast,v)
            neff = get_eff_index(self.wl,w)
            if len(neffs)<4:
                neffs.append(neff)
                zs.append(z)
                vs.append(v)
                vlast = v
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                if z == zf:
                    break
                z = min(zf,z+zstep0)
                continue
            
            # interpolate
            neff_interp = interp1d(zs[-4:],np.array(neffs)[-4:,:],kind='cubic',axis=0,fill_value="extrapolate")
            err = np.sum(np.abs(neff-neff_interp(z)))
            if err < tol or zstep0 == min_zstep:
                neffs.append(neff)
                zs.append(z)
                vs.append(v)
                vlast = v
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                if z == zf:
                    break
                if err < 0.1 * tol:
                    zstep0*=2
                z = min(zf,z+zstep0)
            else:
                print("\rcurrent z: {0} / {1}; tol. not met, reducing step        ".format(z,zf),end='',flush=True)   
                z -= zstep0 
                zstep0 = max(zstep0/2,min_zstep)
                z += zstep0
        
        neffs = np.array(neffs)
        vs = np.array(vs)
        neff_funcs = []
        zs = np.array(zs)
        for i in range(self.Nmax):
            neff_funcs.append(UnivariateSpline(zs,neffs[:,i],s=0))
        self.neffs_funcs = neff_funcs
        self.neffs = neffs
        self.zs = zs
        self.vs = vs # these modes may be inaccurate if there are degeneracies
        print("time elapsed: ",time.time()-start_time)
        return zs , neffs
    
    def compute_modes(self,zi=None,zf=None,mesh=None,tag='',save=False):
        """ compute the modes through the waveguide, using an adaptive stepping scheme. this requires
        greater accuracy in z than get_neffs() since mode shapes may change rapidly even if
        eigenvalues do not.

        ARGS:
            zi: initial z coordinate, if None zi=0
            zf: final z coordinate. if None zf is the waveguide length
            mesh: (optional) a mesh object if you don't want to use the auto-generated, default mesh
            tag: a string identifier which will be attached to any saved files
            save: set True to save function output (z values, effective indices, modes)
        RETURNS:
            zs: array of z coordinates
            neffs: array of effective indices computed on zs
            vs:modes
        """
        zi = 0 if zi is None else zi

        if zf is None:
            assert self.wvg.z_ex is not None, "loaded waveguide has no set length (attribute z_ex), pass in a value for zf."
            zf = self.wvg.z_ex

        if zi==zf:
            return self.solve_at(zi)

        tol = self.zstep_tol
        fixed_step = self.fixed_zstep
        min_zstep = self.min_zstep
        max_zstep = self.max_zstep

        zstep0 = 10 if fixed_step is None else fixed_step # starting step

        self.wvg.update(0) # always initialize at 0, and adjust to any non-zero zi values
        neffs = []
        vs = []
        vs_dec = [] # decimated version of vs, used to control adaptive stepping
        zs = []

        meshpoints = []

        ps = "_"+tag if tag is not None else ""

        if save:
            meshwriteto=self.save_dir+"/meshes/mesh"+ps
        else:
            meshwriteto=None
        
        mesh = self.generate_mesh(writeto=meshwriteto)
        _mesh = copy.deepcopy(mesh)
        IOR_dict = self.wvg.assign_IOR()
        z = zi
        
        print("computing modes ...")
        while True:
            self.wvg.transform_mesh(mesh,0,z,_mesh)
            w,v,N = solve_waveguide(_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            if len(vs)>0:
                v,w = self.swap_modes(vs[-1],v,w)
                for gr in self.degen_groups:
                    self.correct_degeneracy(gr,vs[-1],v)
                self.make_sign_consistent(vs[-1],v)
            # produce an average down version of modes to control adaptive stepping
            vdec = self.decimate(v)
            neff = get_eff_index(self.wl,w)
            if len(neffs)<4:
                neffs.append(neff)
                zs.append(z)
                vs.append(v)
                vs_dec.append(vdec)
                if self.cmat_correction_mode == "from_interp":
                    meshpoints.append(np.copy(_mesh.points[:,:2]))

                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                if z == zf:
                    break
                z = min(zf,z+zstep0)
                continue
            else:
                # interpolate the modes
                vinterp = CubicSpline(np.array(zs[-4:]), vs_dec[-4:] ,axis=0)
                err = np.sum(np.abs(vdec-vinterp(z)))
    
                if err < tol or zstep0 == min_zstep:
                    neffs.append(neff)
                    zs.append(z)
                    vs.append(v)
                    vs_dec.append(vdec)
                    if self.cmat_correction_mode == "from_interp":
                        meshpoints.append(np.copy(_mesh.points[:,:2]))
                    print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                    if z == zf:
                        break
                    if err < 0.1 * tol:
                        zstep0 = min(max_zstep,zstep0*2)
                    z = min(zf,z+zstep0)
                else:
                    print("\rcurrent z: {0} / {1}; tol. not met, reducing step        ".format(z,zf),end='',flush=True)   
                    z = zs[-1] 
                    zstep0 = max(zstep0/2,min_zstep)
                    z = min(zf,z+zstep0)
        
        neffs = np.array(neffs)
        vs = np.array(vs)
        zs = np.array(zs)
        self.neffs = neffs
        self.zs = zs
        self.vs = vs
        self.mesh = mesh
        self.meshpoints = np.array(meshpoints)
        if save:
            self.save(zs,None,neffs,vs,self.meshpoints,tag=tag)
        self.make_interp_funcs(zs,True,False,True)
        return zs,neffs,vs

    def compute_cmats(self,zs=None,vs=None,mesh=None,tag='',save=False):
        """ compute the coupling coefficient matrices.

        ARGS:
            zs: array of z values for the mode array (next arg). if None, use self.zs
            vs: array of eigenmodes
            mesh: finite element mesh of the modes
            tag: a string identifier which will be attached to any saved files
            save: set True to save function output (z values, effective indices, modes)

        RETURNS:
            cmats: coupling coefficient matrices calculated at each z value
        """
        zs = self.zs if zs is None else zs
        vs = self.vs if vs is None else vs
        mesh = self.mesh if mesh is None else mesh
        _mesh = copy.deepcopy(mesh)
        zi,zf = zs[0],zs[-1]
        print("\ncomputing coupling matrix ...")
        vi = CubicSpline(zs,vs,axis=0)

        dmeshdz = None
        if self.cmat_correction_mode == "from_interp":
            points = CubicSpline(zs,self.meshpoints,axis=0)
            dmeshdz = points.derivative()

        dvdz = vi.derivative()
        cmats = []
        points0 = mesh.points.T[:2,:]
        for i,z in enumerate(zs):
            print("\rcurrent z: {0} / {1}        ".format(z,zf),end='',flush=True)
            # the derivative measured by the interpolant comoves with mesh points
            # need to subtract out the x,y component to isolate partial z derivative

            if self.cmat_correction_mode == "from_interp":
                dxydz = dmeshdz(z)
                _mesh.points = points(z)
            else:
                self.wvg.transform_mesh(mesh,0,z,_mesh)
                dxydz = np.array(self.wvg.deriv_transform(points0[0],points0[1],0,z)).T
            
            B = construct_B(_mesh,True)
            dvdxy = FEval.transverse_gradient(vs[i],_mesh.cells[1].data,_mesh.points)
            cor = np.sum(dxydz[None,:,:]*dvdxy,axis=2)
            cmat = self.inner_product(dvdz(z)-cor,vs[i],B)
            cmats.append(cmat)
        cmats = np.array(cmats)  
        self.cmats = cmats   
        self.make_interp_funcs(make_cmat=True)
        if save:
            self.save(cmats=cmats,tag=tag) 
        return cmats

    def make_sign_consistent(self,v,_v):
        """ alter the eigenmodes _v, assumed to be similar to v, so that they 
        have consistent overall sign with v. """

        flip_mask = np.sum(np.abs(v-_v),axis=1) > np.sum(np.abs(v+_v),axis=1)
        _v[flip_mask] *= -1
        return flip_mask
    
    def inner_product(self,v1,v2,B):
        """ compute the inner product (or overlap integral) between fields v1 & v2, which are on the same mesh.
            this requires mesh information, either by passing in the mesh explicitly or by 
            passing in the B matrix. this is the matrix on the RHS of the generalized
            eigenvalue problem and can be calculated e.g. with wavesolve.FEsolver.construct_B()

            the inner product is v1.T * B * v2 , where * is matrix mul.
        """
        # this is just v1^T B v2
        # using dot() (as opposed to np.tensordot) gives a speedup when B is sparse
        return B.dot(v1.T).T.dot(v2.T)

    #endregion

    #region i/o utility

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
        if not os.path.exists(self.save_dir+'/meshpoints'):
            os.makedirs(self.save_dir+'/meshpoints')

    def save(self,zs=None,cmats=None,neffs=None,vs=None,meshpoints=None,tag=""):
        ps = "" if tag == "" else "_"+tag
        if vs is not None:
            vs = np.array(vs) # eigenmode array is (KxNxM) for M mesh points, N eigenmodes, and K z values
            np.save(self.save_dir+'/eigenmodes/eigenmodes'+ps,vs)
        if cmats is not None:
            np.save(self.save_dir+'/cplcoeffs/cplcoeffs'+ps,cmats)
        if neffs is not None:
            np.save(self.save_dir+'/eigenvalues/eigenvalues'+ps,neffs)
        if zs is not None:
            np.save(self.save_dir+'/zvals/zvals'+ps,zs)
        if meshpoints is not None and len(meshpoints)>0:
            np.save(self.save_dir+'/meshpoints/meshpoints'+ps,meshpoints)

    def load(self,tag=""):
        ps = "" if tag == "" else "_"+tag

        self.neffs = np.load(self.save_dir+'/eigenvalues/eigenvalues'+ps+'.npy')
        self.vs = np.load(self.save_dir+'/eigenmodes/eigenmodes'+ps+".npy")
        self.zs = np.load(self.save_dir+'/zvals/zvals'+ps+'.npy')

        try:
            self.cmats = np.load(self.save_dir+'/cplcoeffs/cplcoeffs'+ps+'.npy')
            self.make_interp_funcs(self.zs,make_neff=False,make_v=False)
        except:
            print("no coupling matrix file found ... skipping")
            pass

        try: 
            self.meshpoints = np.load(self.save_dir+'/meshpoints/meshpoints'+ps+".npy")
        except:
            pass
            
        self.mesh = load_meshio_mesh(self.save_dir+'/meshes/mesh'+ps)
        self.points0 = np.copy(self.mesh.points)

        if self.Nmax==None:
            self.Nmax = self.neffs.shape[1]

        self.make_interp_funcs(make_cmat=False)
        
    
    #endregion

    #region aux funcs

    def make_interp_funcs(self,zs=None,make_neff=True,make_cmat=True,make_v=True):
        """ construct interpolation functions for coupling matrices and mode effective indices,
            loaded into self.cmats and self.neffs, which were computed on an array of z values self.zs.
        """
        
        if zs is None:
            zs = np.copy(self.zs)
        
        if make_cmat:
            def make_c_func(i,j):
                assert i < j, "i must be < j in make_interp_funcs()"
                return UnivariateSpline(zs,0.5*(self.cmats[:,i,j]-self.cmats[:,j,i]),ext=0,s=0)
            cmat_funcs = []
            for j in range(1,self.Nmax):
                for i in range(j):
                    cmat_funcs.append(make_c_func(i,j))
            
            self.cmats_funcs = cmat_funcs
        
        if make_neff:
            neff_funcs = []
            for i in range(self.Nmax):
                neff_funcs.append(UnivariateSpline(zs,self.neffs[:,i],s=0))
            
            self.neffs_funcs = neff_funcs
            self.neffs_int_funcs = [neff_func.antiderivative() for neff_func in neff_funcs]
            self.neffs_dif_funcs = [neff_func.derivative() for neff_func in neff_funcs]

        if make_v:
            self.get_v = CubicSpline(zs,self.vs,axis=0)

    def get_cmat(self,z):
        """ using interpolation, compute the cross-coupling matrix at z """
        out = np.zeros((self.Nmax,self.Nmax))
        k = 0
        for j in range(1,self.Nmax):
            for i in range(j):
                val = self.cmats_funcs[k](z)
                out[i,j] = -val
                out[j,i] = val
                k+=1
        return out
    
    def get_neff(self,z):
        """ using interpolation, compute the array of mode effective indices at z """
        return np.array([neff(z) for neff in self.neffs_funcs])
    
    def get_int_neff(self,z):
        """ compute the antiderivative of the mode effective indices at z"""
        return np.array([neffi(z) for neffi in self.neffs_int_funcs])

    def get_dif_neff(self,z):
        return np.array([neffd (z) for neffd in self.neffs_dif_funcs]) 

    def WKB_cor(self,z):
        dbeta_dz = self.k * self.get_dif_neff(z) 
        return -0.5 * dbeta_dz / self.get_neff(z)

    def compute_change_of_basis(self,newbasis,z=None,u=None):
        """ compute the (N x N) change of basis matrix between the current N-dimensional eigenbasis at z and a new basis 
        
        ARGS: 
            newbasis: MxN array of N eigenmodes computed over M mesh points, which we want to expand in
            z: z coordinate at which the currently loaded eigenbasis should be evaluated
            u: (optional) Nx1 modal vector to express in new basis

        RETURNS:
            cob, _u : Nmax x Nmax change of basis matrix and the vector u in the new basis, if u was provided.
            
        """
        if z is None:
            z = self.zs[-1]

        m = self.make_mesh_at_z(z)
        B = construct_B(m,sparse=True)
        oldbasis = self.get_v(z)
        cob = self.inner_product(newbasis,oldbasis,B)
        self.channel_basis_matrix = cob
        if u is not None:
            return cob,np.dot(cob,u)
        return cob
    
    def compute_isolated_basis(self,z=None):
        """ compute the eigenbasis corresponding to "isolated" channels of the waveguide.
        this only makes sense for waveguides such as PhotonicLantern, Dicoupler, and Tricoupler.

        ARGS:
            z: z coordinate of eigenbasis. if None, this defaults to the end of the waveguide.

        RETURNS:
            (array): the new basis
        """
        if z is None:
            z = self.zs[-1]

        m = self.make_mesh_at_z(z)
        self.wvg.assign_IOR() 
        wvg_dim = len(self.wvg.prim3Dgroups[-1])
        _v = np.zeros((wvg_dim,m.points.shape[0])) # array to store the new basis

        for i in range(wvg_dim):
            _dict = self.wvg.isolate(i) # use the PhotonicLantern.isolate() function to make a new dictionary of refractive index values which isolates a single core
            #print(_dict)
            _wi,_vi,_Ni = solve_waveguide(m,self.wl,_dict,sparse=True,Nmax=1) # then pass the new dictionary into the eigenmode solver
            if np.sum(_vi) < 0: # attempt consistent sign
                _vi *= -1
            _v[i,:] = _vi # save the "port" eigenmode into _v

        return _v

    def to_channel_basis(self,uf,z=None):   
        """ convert mode amplitude vector to basis of channel eigenmodes. 
        this only makes sense if the waveguide has defined output channels.

        ARGS:
            uf: mode amplitude vector (phase factored *in*)
            z: z coordinate. if None, z is set to the end of the waveguide.
        RETURNS:
            (vector): the vector of mode amplitudes in output channel basis.
        """
        
        if self.channel_basis_matrix is None:  
            _v = self.compute_isolated_basis(z)
            self.compute_change_of_basis(_v,z)
        return np.dot(self.channel_basis_matrix,uf)


    def make_field(self,mode_amps,z,plot=False,apply_phase=True):
        """ construct the finite element field corresponding to the modal vector u and the eigenbasis at z. 
        
        ARGS:
            mode_amps: the modal vector expressing the field (with the fast e^{i beta z}) phase oscillation factored
            z: the z coordinate corresponding to u
            plot (opt.): set True to auto-plot the norm of the field. 
            apply_phase (bool): whether the :math:`e^{i\\beta_j z}` phase factor
                                needs to be applied to mode_amps; default True.
        
        RETURNS:
            (array): the field evaluated on the mesh nodes.
        """
        u = np.array(mode_amps,dtype=np.complex128)
        basis = self.get_v(z)
        zi = self.zs[0]
        if apply_phase:
            phase = np.exp(1.j*self.k*np.array(self.get_int_neff(z)-self.get_int_neff(zi)))
            uf = u*phase
            field = np.sum(uf[:,None]*basis,axis=0)
        else:
            field = np.sum(u[:,None]*basis,axis=0)
        if plot:
            self.plot_cfield(field,z,show_mesh=True)
        return field

    def decimate(self,arr,outsize=10,axis=1):
        #default is really avg down to 10 vals, not decimation
        split_arrs = np.array_split(arr,outsize,axis=axis)
        return np.array([np.mean(a,axis=axis) for a in split_arrs]).T

    def swap_modes(self,v,_v,_w):
        """ swap modes in _v so they match their arrangement in v - useful for tracking
        modes through eigenvalue crossings.

        ARGS:
            v: the mode basis we want to match
            _v: the mode basis we want to swap
            _w: the eigenvalues of _v, which also need to be swapped.
        
        RETURNS:
            (tuple): a tuple containing:
                - the swapped mode array
                - the swapped eigenvalue array
        """
        indices = np.zeros(self.Nmax,dtype=int)
        for i in range(1,self.Nmax):
            if i in indices: # shortcut ...
                continue
            max_overlap = 0
            _i = i
            for j in range(i,self.Nmax):
                overlap = np.abs(np.sum(v[i]*_v[j]))
                if overlap > max_overlap:
                    max_overlap = overlap
                    _i = j
            if i != _i:
                #print("swapping ...",(i,_i))
                indices[i] = _i
                indices[_i] = i
                #print(indices)
            else:
                indices[i] = _i
        return _v[indices],_w[indices]

    def correct_degeneracy(self,group,v,_v,q=None):
        """ used least-squares minimization to transform modes _v so that they match v. mutates _v
        
        ARGS:
            group: a list of mode indexes setting which modes are to be considered degenerate.
            v: the mode basis we want to match
            _v: the mode basis we want to transform. We are allowed to apply a unitary transformation over the 
                modes specified by group.
            q (opt.): if not None, q is taken as the transformation matrix. if None, q is computed through lsq min.
        
        RETURNS:
            (tuple): a tuple containing:
                - v: the original mode basis
                - _v: the transformed mode basis
                - q: the change-of-basis matrix
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
        
    #region mesh gen

    def generate_mesh(self,writeto=None):
        """ generate a mesh for the loaded waveguide according to class attributes.
        
        ARGS:
            writeto (str or None): set to a str (the filename) to write the mesh to file. 
        """
        return self.wvg.make_mesh(writeto=writeto)

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

    def make_mesh_at_z(self,z):
        """ make the mesh corresponding to the waveguide cross-section at z. 
        
        ARGS:
            z: z coordinate for the mesh.
        
        RETURNS:
            (meshio mesh): the corresponding mesh at z.
        """
        mesh = self.generate_mesh()
        return self.wvg.transform_mesh(mesh,0,z)

    #endregion

    #region plotting

    def plot_neffs(self):
        """ plot the effective indices of the eigenmodes """
        neffs = self.neffs.T
        for i in range(self.Nmax):
            plt.plot(self.zs,neffs[i],label="mode "+str(i),)
        plt.xlabel(r"$z$")
        plt.ylabel("effective index")
        plt.legend(loc='best')
        plt.show()
    
    def plot_neff_diffs(self):
        """ plot the difference between the effective index of each mode and that of the fundamental, on y-semilog """
        neffs = self.neffs.T
        for i in range(1,self.Nmax):
            plt.semilogy(self.zs,neffs[0]-neffs[i],label="mode "+str(i),)
        plt.xlabel(r"$z$")
        plt.ylabel("difference in index from mode 0")
        plt.legend(loc='best')
        plt.show()   

    def plot_field(self,field,z=None,mesh=None,ax=None,show_mesh=False):
        """ plot a real-valued finite element field, evaluated on the points of the mesh corresponding to the waveguide cross-section at z.
        
        ARGS:
            field: the finite element field to plot, e.g. the output of make_field() or an eigenmode (a row of self.vs)
            z: the z coordinate corresponding to field
            mesh (opt.): the mesh object defining the points that field is evaluated on. If None, a mesh will be auto-generated
            ax (opt.): a matplotlib axis where the plot should go. if not None, you will need to call matplotlib.pyplot.show() manually
            show_mesh (opt.): whether or not to draw the mesh in the plot
        """
        mesh = self.make_mesh_at_z(z) if mesh is None else mesh
        plot_field(field,mesh,ax,show_mesh)

    def plot_coupling_coeffs(self):
        """ plot coupling coefficient matrix vs z values. """

        fig,ax = plt.subplots()

        colors = ['#377eb8', '#ff7f00', '#4daf4a',
                        '#f781bf', '#a65628', '#984ea3',
                        '#999999', '#e41a1c', '#dede00']
        line_styles = ['solid','dashed','dotted','dashdot',(5,(10,3))]

        for j in range(self.Nmax): 
            for i in range(j):
                ax.plot(self.zs,self.cmats[:,i,j],label=str(i)+str(j),ls=line_styles[i%5],c=colors[j%9])

        for z in self.zs: # plot vertical bars at every z value.
            ax.axvline(x=z,alpha=0.1,color='k',zorder=-100)
        ax.legend(bbox_to_anchor=(1., 1.))
        ax.set_title("coupling coefficient matrix")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\kappa_{ij}$")
        plt.tight_layout()
        plt.show()

    def plot_mode_powers(self,zs,us):
        """ plot mode powers against :math:`z`.

        ARGS:
            zs: array of :math:`z` values.
            us: array of mode amplitudes, e.g. from propagate(). the first axis corresponds to z.
        """
        for i in range(us.shape[0]):
            plt.plot(zs,np.power(np.abs(us[i]),2),label="mode 0")     
        plt.xlabel(r'$z$ (um)')
        plt.ylabel("power")
        plt.legend(loc='best')   
        plt.show()   
            
    def plot_cfield(self,field,z=None,mesh=None,fig=None,ax=None,show_mesh=False,res=50,tree=None):
        """ plot a complex-valued field evaluated a finite element mesh.
        
        ARGS:
            field: the complex field to be plotted
            z: the z value along the waveguide corresponding to field. alternatively, explicitly pass the mext arg
            mesh: the finite element mesh corresponding to field
            fig, ax: matplotlib figure and axis objects; if None, a new fig,ax is created
            show_mesh: shows the edges of the FE mesh in the plot
            res: the field will be evaluated on an res x res grid to form the plot
            tree: explicitly pass in the BVH tree for the FE mesh (you don't need to do this unless you want more speed)
        """
        mesh = self.make_mesh_at_z(z) if mesh is None else mesh
        plot_cfield(field,mesh,fig,ax,show_mesh,res,tree)

    def plot_waveguide_mode(self,idx,fig=None,ax=None):
        """ plot the real-valued eigenmode <idx> of the current waveguide, from modes saved in self.vs .
        this plot comes with a z-slider, which can be moved to see how mode changes along the guide.
        
        ARGS:
            idx: the index of the mode to be plotted (mode 0,1,2...)
            fig: matplotlib figure object; if None, one will be made
            ax: matplotlib axis objects; if None, one will be made
            animate (bool); set True to animate through z
        
        RETURNS:
            (matplotlib.widgets.slider): the slider object. a reference to the slider needs to be kept to prevent garbage collection from removing it.
        """
        plot=False
        if ax is None or fig is None:
            fig,ax = plt.subplots(1,1)
            fig.subplots_adjust(bottom=0.25)
            plot=True

        triangulation = Triangulation(self.mesh.points[:,0],self.mesh.points[:,1],self.mesh.cells[1].data[:,:3])
        mesh = copy.deepcopy(self.mesh)
        contourf = ax.tricontourf(triangulation,self.vs[0,idx],levels=40)
        ax.set_aspect('equal')
        x0,y0,w,h = ax.get_position().bounds
        axsl = fig.add_axes([x0,y0-0.25,w,0.1])
        slider = Slider(ax=axsl,label=r'$z$',valmin=self.zs[0],valmax=self.zs[-1],valinit=self.zs[0])
        def update(z):
            ax.clear()
            v = self.get_v(z)[idx]
            self.wvg.transform_mesh(self.mesh,0,z,mesh)
            x = mesh.points[:,0]
            y = mesh.points[:,1]
            contourf = ax.tricontourf(Triangulation(x,y,triangulation.triangles),v,levels=40)
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        if plot:
            plt.show()
        return slider # a reference to the slider needs to be preserved to prevent gc :/
    #endregion
