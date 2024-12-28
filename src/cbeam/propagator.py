from __future__ import annotations
import numpy as np,copy,time,os
from wavesolve.fe_solver import solve_waveguide,get_eff_index,construct_B,plot_eigenvector
from cbeam.waveguide import load_meshio_mesh,Waveguide,plot_mesh
from scipy.interpolate import UnivariateSpline,interp1d,CubicSpline
from cbeam import FEval
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation,CubicTriInterpolator,LinearTriInterpolator
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from typing import Union
from bisect import bisect_left

normcmap = np.zeros([256, 4])
normcmap[:, 3] = np.linspace(0, 1, 256)[::-1]
normcmap = ListedColormap(normcmap)

def plot_cfield(field,mesh,fig=None,ax=None,show_mesh=False,res=1.,xlim=None,ylim=None):
    show = False
    if ax is None:
        fig,ax = plt.subplots(1,1)
        show = True

    xm = np.max(mesh.points[:,0])
    ym = np.max(mesh.points[:,1])
    xlim = (-xm,xm) if xlim is None else xlim
    ylim = (-ym,ym) if ylim is None else ylim
    xa = np.arange(*xlim,res,dtype=np.float64)
    ya = np.arange(*ylim,res,dtype=np.float64)
    if not hasattr(mesh,'tree'):
        FEval.sort_mesh(mesh)
    fgrid = np.array(FEval.evaluate_grid(xa,ya,field,mesh.tree)).T
    alphas = np.abs(fgrid)
    alphas /= np.max(alphas)
    ax.set_facecolor('k')
    im = ax.imshow(np.angle(fgrid),extent=(*xlim,*ylim),cmap="hsv",vmin=-np.pi,vmax=np.pi,origin="lower")
    ax.imshow(alphas,extent=(*xlim,*ylim),cmap=normcmap,interpolation="bicubic",origin="lower")

    if show_mesh:
        plot_mesh(mesh,plot_points=False,ax=ax,alpha=0.1,verbose=False)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
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

    #: float: controls adaptive stepping. higher = more steps (more accurate).
    #: this is base 10 logarithmic. if you want to reduce tolerance by 10x, set to 1, etc. 
    z_acc = 0.
    #: float or None: set to a numeric value to use a fixed zstep
    fixed_zstep = None     
    #: float: minimum z step value when computing modes.
    min_zstep = 0.625
    #: float: minimum z step value when computing effective indices
    min_zstep_neff = 10. 
    #: float: maximum z step value when computing modes or effective indices
    max_zstep = np.inf
    #: float: starting z step
    init_zstep = 10.

    # misc params
    degen_crit = 1e-5         # the minimum difference in effective index two modes can have before they are considered degenerate    
    
    degen_groups = []

    #: list: used to zero certain modes during calculations. useful if there's a cladding mode
    #: which is behaving erratically and slowing down the adaptive stepping.
    skipped_modes = []

    #: bool: whether the propagator is allowed to swap modes (to track them through eigenvalue crossings)
    allow_swaps = True

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
        self.mesh = None

        self.channel_basis_matrix = None

        if save_dir is None:
            self.save_dir = './data'
        else:
            self.save_dir = save_dir

        self.check_and_make_folders()                
    
    #region main functions

    def solve_at(self,z=0,mesh=None):
        """ solve for waveguide modes at given z value. returns effective indices and eigenmodes.
        also stores the finite element mesh used to self.mesh
        
        ARGS:
            z (float): z coordinate along the waveguide for the solve, default 0
            mesh: FE mesh to solve on. if None, we make a mesh corresponding to the given z
        
        RETURNS:
            (tuple) : a tuple containing:

                - neff: an array of effective indices
                - v: an array of eigenmodes, dimensions NxM for N modes and M mesh points
        """
        mesh = self.make_mesh_at_z(z) if mesh is None else mesh
        IOR_dict = self.wvg.assign_IOR()
        w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
        self.mesh = mesh
        return get_eff_index(self.wl,w),v

    def characterize(self,zi=None,zf=None,mesh=None,tag='',save=False):
        """ compute the modes and coupling coefficients of the currently loaded waveguide,
        and set things up for propagation. for reference, this function calls compute_modes() 
        and then compute_cmat().
        
        ARGS:
            zi: initial z coordinate, default 0
            zf: final z coordinate. if None, use the waveguide length
            mesh: an initial mesh (z=zi) if you don't want to use the auto-generated, default mesh
            tag: a string identifier which will be attached to any saved files
            save: set True to save function output (z values, effective indices, modes)
        
        RETURNS:
            (tuple) : a tuple containing:

                - zs: array of z coordinates
                - neffs: array of effective indices computed on zs; 0th axis is "z axis"
                - vs: eigenmodes computed on zs; 0th axis is "z axis"
                - cmats: coupling coefficients computed on zs; 0th axis is "z axis"
        """
        
        # z invariant wvg handling
        if self.wvg.z_invariant:
            ps = "_"+tag if tag is not None else ""
            if save:
                meshwriteto=self.save_dir+"/meshes/mesh"+ps
            else:
                meshwriteto=None

            self.mesh = self.generate_mesh(writeto=meshwriteto) if mesh is None else mesh
            print("mesh has ",len(self.mesh.points)," points")
            neff,v = self.solve_at(0)
            zs = np.array([0.])
            neffs = np.array([neff])
            vs = np.array([v])
            self.zs,self.neffs,self.vs = zs,neffs,vs
            if save:
                self.save(zs=zs,neffs=neffs,vs=vs,tag=tag)
            self.make_interp_funcs_zinv()
            return zs,neffs,vs,None

        # normal handling
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
        assert self.zs is not None, "no propagation data detected ... run characterize() or load() first"
        if zi is None:
            zi = self.zs[0]
        if zf is None:
            zf = self.zs[-1]

        # z invariant case
        if len(self.zs) == 1:
            return self.zs,np.array([u0]),self.apply_phase(u0,zf,zi)

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
        return sol.t,sol.y.T,uf

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
    def compute_neffs(self,zi=0,zf=None,mesh=None,tag='',save=False):
        """ compute the effective refractive indices through a waveguide, using an adaptive step scheme. also saves interpolation functions to self.neffs_funcs
        
        ARGS:
            zi: initial z coordinate, default 0
            zf: final z coordinate. if None, use waveguide's length
            mesh: a mesh object if you don't want to use the auto-generated, default mesh
            tag: string identifier to attach to saved files for computation results (if saving)
            save: whether or not to save the computation results
        
        RETURNS:
            (tuple): a tuple containing:
                - zs: array of z coordinates
                - neffs: array of effective indices computed on zs
        """
        zf = self.wvg.z_ex if zf is None else zf
        start_time = time.time()
        neffs = []
        zs = []
        vs = []
        self.wvg.update(0)
        ps = "_"+tag if tag is not None else ""
        if save:
            meshwriteto=self.save_dir+"/meshes/mesh"+ps
        else:
            meshwriteto=None
        if mesh is None:
            mesh = self.generate_mesh(meshwriteto) # use default vals
        self.mesh = mesh
        print("mesh has ",len(self.mesh.points)," points")
        zstep0 = self.init_zstep if self.fixed_zstep is None else self.fixed_zstep
        min_zstep = self.min_zstep_neff
        neff_interp = None

        IOR_dict = self.wvg.assign_IOR()
        z = zi
        
        print("computing effective indices ...")

        if zi==zf:
            w,v,N = solve_waveguide(mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            return z , get_eff_index(self.wl,w) , v

        while True:
            _mesh = self.wvg.transform_mesh(mesh,0,z)
            w,v,N = solve_waveguide(_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
            neff = get_eff_index(self.wl,w)
            if len(neffs)>0:
                if len(neffs) == 1:
                    self.track_modes(vs[-1],v,neffs[-1],neff)
                elif 1<len(vs)<4:
                    N = len(neffs)
                    neff_interp = interp1d(zs[-N:],np.array(neffs)[-N:,:],kind=N-1,axis=0,fill_value="extrapolate")
                    self.track_modes(vs[-1],v,neff_interp(z),neff)
                else:
                    #neff_interp = interp1d(zs[-4:],np.array(neffs)[-4:,:],axis=0,kind=3,fill_value='extrapolate',assume_sorted=True)
                    neff_interp = CubicSpline(zs[-4:],neffs[-4:],axis=0)
                    self.track_modes(vs[-1],v,neff_interp(z),neff)

            if len(neffs)<4 or self.fixed_zstep:
                N = len(neffs)
                neffs.append(neff)
                zs.append(z)
                vs.append(v)
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                if z == zf:
                    break
                z = min(zf,z+zstep0)
                continue
            
            refac=self._ref_fac_n(neff_interp(z),neff,neffs[-1])
            if refac >= 0 or zstep0 == min_zstep:
                neffs.append(neff)
                zs.append(z)
                vs.append(v)
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                if z == zf:
                    break
                if refac == 1:
                    zstep0*=2
                z = min(zf,z+zstep0)
            else:
                print("\rcurrent z: {0} / {1}; tol. not met, reducing step        ".format(z,zf),end='',flush=True)   
                z = zs[-1]
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
        if save:
            self.save(zs,None,neffs,vs,tag=tag)
        self.make_interp_funcs(make_cmat=False)
        print("time elapsed: ",time.time()-start_time)
        return zs , neffs

    def compute_modes(self,zi=None,zf=None,mesh=None,tag='',save=False):
        """ compute the modes through the waveguide, using an adaptive stepping scheme. this requires
        greater accuracy in z than get_neffs() since mode shapes may change rapidly even if
        eigenvalues do not. stores initial mesh to self.mesh

        ARGS:
            zi: initial z coordinate, if None zi=0
            zf: final z coordinate. if None, zf is the waveguide length
            mesh: an initial mesh (z=zi) if you don't want to use the auto-generated, default mesh
            tag: a string identifier which will be attached to any saved files
            save: set True to save function output (z values, effective indices, modes)
            init_modes: an initial mode basis to use for the computation
        RETURNS:
            zs: array of z coordinates
            neffs: array of effective indices computed on zs
            vs:modes
        """
        zi = 0 if zi is None else zi
        hit_min_zstep = False
        if zf is None:
            assert self.wvg.z_ex is not None, "loaded waveguide has no set length (attribute z_ex), pass in a value for zf."
            zf = self.wvg.z_ex

        fixed_step = self.fixed_zstep
        min_zstep = self.min_zstep
        max_zstep = self.max_zstep

        zstep0 = self.init_zstep if fixed_step is None else fixed_step # starting step

        self.wvg.update(0) # always initialize at 0, and adjust to any non-zero zi values
        neffs = []
        vs = []
        vs_dec = [] # decimated version of vs, used to control adaptive stepping
        zs = []
        vinterp = None

        meshpoints = []

        ps = "_"+tag if tag is not None else ""

        if save:
            meshwriteto = self.save_dir+"/meshes/mesh"+ps
        else:
            meshwriteto = None            

        mesh = self.generate_mesh(writeto=meshwriteto) if mesh is None else mesh
        print("mesh has ",len(mesh.points)," points")
        _mesh = copy.deepcopy(mesh)
        IOR_dict = self.wvg.assign_IOR()
        z = zi
        print("computing modes ...")
        while True:
            self.wvg.transform_mesh(mesh,0,z,_mesh)
            if len(vs)==0 and self.vs is not None and self.neffs is not None:
                neff,v = self.neffs[0],self.vs[0]
            else:
                w,v,N = solve_waveguide(_mesh,self.wl,IOR_dict,sparse=True,Nmax=self.Nmax)
                neff = get_eff_index(self.wl,w)            
            if len(neffs)>0:
                if len(neffs) == 1:
                    self.track_modes(vs[-1],v,neffs[-1],neff)
                elif 1<len(vs)<4:
                    N = len(neffs)
                    neff_interp = interp1d(zs[-N:],np.array(neffs)[-N:,:],kind=N-1,axis=0,fill_value="extrapolate")
                    self.track_modes(vs[-1],v,neff_interp(z),neff)
                else:
                    neff_interp = CubicSpline(zs[-4:], neffs[-4:], axis=0, extrapolate=True,bc_type='natural')
                    self.track_modes(vs[-1],v,neff_interp(z),neff)
            
            for gr in self.degen_groups:
                self.avg_degen_neff(gr,neff)
            
            v[self.skipped_modes] = 0.

            # produce an average down version of modes to control adaptive stepping
            vdec = self.decimate(v)
            if len(vs)<4 or fixed_step: # accept the first four computations
                neffs.append(neff)
                zs.append(z)
                vs.append(v)
                vs_dec.append(vdec)
                if self.cmat_correction_mode == "from_interp" and not self.wvg.linear:
                    meshpoints.append(np.copy(_mesh.points[:,:2]))

                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                if z == zf:
                    break
                z = min(zf,z+zstep0)
                continue
            else:
                # interpolate the modes
                vinterp = CubicSpline(np.array(zs[-4:]), vs_dec[-4:] ,axis=0)
                refac = self._ref_fac_v(vdec,vs_dec[-1],vinterp(z))
                if refac>=0 or zstep0 == min_zstep:
                    if not hit_min_zstep and zstep0 == min_zstep:
                        hit_min_zstep = True
                    neffs.append(neff)
                    zs.append(z)
                    vs.append(v)
                    vs_dec.append(vdec)
                    if self.cmat_correction_mode == "from_interp" and not self.wvg.linear:
                        meshpoints.append(np.copy(_mesh.points[:,:2]))
                    print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(z,zf,zstep0),end='',flush=True)
                    if z == zf:
                        break
                    if refac==1:
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
        if self.wvg.linear:
            meshpoints.append(mesh.points[:,:2])
            meshpoints.append(_mesh.points[:,:2])
        self.meshpoints = np.array(meshpoints)
        if save:
            self.save(zs,None,neffs,vs,self.meshpoints,tag=tag)
        self.make_interp_funcs(zs,True,False,True)
        if hit_min_zstep:
            print("\nwarning: hit minimum z step when computing modes")
        return zs,neffs,vs

    def load_init_conds(self,init_prop:Propagator,z=None):
        """ load an eigenmode basis from init_prop as the initial basis for 
        this propagator's calculations.

        ARGS:
            init_prop (Propagator): the Propagator object to load modes from
            z (float,None): the z value of the modes to load from init_prop. if None,
            use the last z value in init_prop.zs .
        """

        self.vs = []
        self.neffs = []
        if z is None:
            self.vs.append(init_prop.vs[-1])
            self.neffs.append(init_prop.neffs[-1])
        else:
            self.vs.append(init_prop.get_v(z))
            self.neffs.append(init_prop.get_neff(z))

    def compute_cmats(self,zs=None,vs=None,mesh=None,tag='',save=False):
        """ compute the coupling coefficient matrices.

        ARGS:
            zs: array of z values for the mode array (next arg). if None, use self.zs
            vs: array of eigenmodes. if None, use self.vs
            mesh: finite element mesh of the modes. if None, use self.mesh
            tag: a string identifier which will be attached to any saved files
            save: set True to save function output (z values, effective indices, modes)

        RETURNS:
            cmats: coupling coefficient matrices calculated at each z value
        """
        zs = self.zs if zs is None else zs
        vs = self.vs if vs is None else vs
        mesh = self.mesh if mesh is None else mesh
        _mesh = copy.deepcopy(mesh)
        if _mesh.points.shape[1] == 3:
            _mesh.points = _mesh.points[:,:2]
        zi,zf = zs[0],zs[-1]
        print("\ncomputing coupling matrix ...")
        vi = CubicSpline(zs,vs,axis=0)

        _linear = (len(self.meshpoints) == 2)

        dmeshdz = None
        if self.cmat_correction_mode == "from_interp":
            if _linear:
                slope = (self.meshpoints[1]-self.meshpoints[0])/(zs[-1]-zs[0])
                points = lambda z: slope*(z-zs[0]) + self.meshpoints[0]
                dmeshdz = lambda z: slope
            else:
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
                if _linear:
                    _mesh.points[:] = points(z)
                else:
                    _mesh.points[:] = self.meshpoints[i]
                dxydz = dmeshdz(z)
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

    def _ref_fac_v(self,v,vlast,vi):
        resids = v-vi
        resids[self.skipped_modes] = 0.
        err = np.sqrt(np.mean(np.power(resids,2)))
        tol = max(np.sqrt(np.mean(np.power(v-vlast,2)))/100.,1e-7) * np.power(10.,-np.float(self.z_acc))
        if err < 0.1*tol:
            return 1
        elif err > tol:
            return -1
        else:
            return 0
    
    def _ref_fac_n(self,ninterp,n,nlast):
        resids = ninterp-n
        resids[self.skipped_modes] = 0.
        err = np.sqrt(np.mean(np.power(resids,2)))
        nsort = sorted(nlast,reverse=True)
        tol = max((nsort[0]-nsort[1])/100.,1e-9) * np.power(10.,-np.float(self.z_acc))
        if 0.1*tol < err < tol:
            return 0
        elif err < 0.1* tol:
            return 1
        else:
            return -1
        
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
        """ load the z values, effective indices, mode profiles, coupling coefficients, and mesh points
        saved to files specified by <tag>.
        """
        ps = "" if tag == "" else "_"+tag
        self.neffs = np.load(self.save_dir+'/eigenvalues/eigenvalues'+ps+'.npy')
        self.vs = np.load(self.save_dir+'/eigenmodes/eigenmodes'+ps+".npy")
        self.zs = np.load(self.save_dir+'/zvals/zvals'+ps+'.npy')
        if self.Nmax is None:
            self.Nmax = len(self.neffs[0])

        try:
            self.cmats = np.load(self.save_dir+'/cplcoeffs/cplcoeffs'+ps+'.npy')
            self.make_interp_funcs(self.zs,make_neff=False,make_v=False)
        except:
            print("no coupling matrix file found ... skipping")
            pass

        try: 
            self.meshpoints = np.load(self.save_dir+'/meshpoints/meshpoints'+ps+".npy")
        except:
            print("no mesh points found ... skipping")
            pass
            
        self.mesh = load_meshio_mesh(self.save_dir+'/meshes/mesh'+ps)
        self.points0 = np.copy(self.mesh.points)

        if self.Nmax==None:
            self.Nmax = self.neffs.shape[1]
        
        if len(self.zs)>1:
            self.make_interp_funcs(make_cmat=False)
        else:
            self.make_interp_funcs_zinv()
        
    
    #endregion

    #region aux funcs

    def make_interp_funcs(self,zs=None,make_neff=True,make_cmat=True,make_v=True):
        """ construct interpolation functions for coupling matrices and mode effective indices,
            loaded into self.cmats and self.neffs, which were computed on an array of z values self.zs.
        
        ARGS:
            zs (None or array): the z values to use for interpolation. if none, use self.zs
            make_neff (bool): set True to make effective interpolation function
            make_cmat (bool): set True to make coupling matrix interpolation function
            make_v (bool): set True to make eigenmode interpolation function
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

    def make_interp_funcs_zinv(self):
        #self.neffs_funcs = [lambda z: neff for neff in self.neffs[0]]
        #self.neffs_int_funcs = [lambda z: z*neff for neff in self.neffs[0]]
        #self.neffs_dif_funcs = [lambda z: 0 for neff in self.neffs[0]]
        self.get_v = lambda z: self.vs[0]
        self.get_neff = lambda z: self.neffs[0]
        self.get_int_neff = lambda z: z*self.neffs[0]
        self.get_dif_neff = lambda z: np.zeros_like(self.neffs[0])
        self.get_cmat = lambda z: np.zeros((self.Nmax,self.Nmax))

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
        this only makes sense if the waveguide has defined output channels
        (i.e. the waveguide must have the isolate() function).

        ARGS:
            uf: mode amplitude vector (phase factored *in*)
            z: z coordinate. if None, z is set to the last value of self.zs
        RETURNS:
            (vector): the vector of mode amplitudes in output channel basis.
        """
        
        if self.channel_basis_matrix is None:  
            _v = self.compute_isolated_basis(z)
            self.compute_change_of_basis(_v,z)
        return np.dot(self.channel_basis_matrix,uf)


    def make_field(self,mode_amps,z=None,plot=False,apply_phase=True):
        """ construct the finite element field corresponding to the modal vector u and the eigenbasis at z. 
        
        ARGS:
            mode_amps: the modal vector expressing the field with :math:`e^{i\beta_j z}` phase oscillation factored out
            z: the z coordinate corresponding to mode_amps
            plot (opt.): set True to auto-plot the norm of the field. 
            apply_phase (bool): whether the :math:`e^{i\\beta_j z}` phase factor
                                needs to be applied to mode_amps; default True.
        
        RETURNS:
            (array): the field evaluated on the mesh nodes.
        """
        assert self.zs is not None, "no propagation data detected ... run characterize() or load() first"
        zinv = len(self.zs) == 1
        assert z is not None or zinv, "z can only be left as None if the waveguide is z-invariant"
        z = 0 if z is None and zinv else z

        zi = self.zs[0] if self.zs is not None else 0.
        u = np.array(mode_amps,dtype=np.complex128)
        basis = self.get_v(z)
        
        if apply_phase:
            uf = self.apply_phase(u,z,zi)
            field = np.sum(uf[:,None]*basis,axis=0)
        else:
            field = np.sum(u[:,None]*basis,axis=0)
        if plot:
            self.plot_cfield(field,z,show_mesh=True)
        return field

    def make_mode_vector(self,field,z=None,mesh=None):
        """ from a field, make a complex mode amplitude vector by decomposing the 
        field into the currently loaded basis. kind of like the opposite of make_field(). 

        ARGS:
            field: the finite element field to decompose into mode amplitudes
            z: the z coordinate of field; default is the first value of self.zs, or 0 if self.zs is not found
        """
        if z is None:
            if self.zs is None:
                z = 0.
            else:
                z = self.zs[0]
        mesh = self.make_mesh_at_z(z) if mesh is None else mesh
        B = construct_B(self.mesh,sparse=True)
        basis = self.get_v(z)
        # take inner product between field and basis modes
        amps = []
        for i in range(basis.shape[0]):
            amps.append(self.inner_product(basis[i],field,B))
        
        return np.array(amps)

    def decimate(self,arr,outsize=10,axis=1):
        #default is really avg down to 10 vals, not decimation
        split_arrs = np.array_split(arr,outsize,axis=axis)
        return np.array([np.mean(a,axis=axis) for a in split_arrs]).T

    def swap_modes(self,w,_w,_v):
        """ permute _w so that it matches w as close as possible.
        permute _v in the same way. """
        # it is mind-blowing that this function works lmao
        # it's so simple ... and confusing
        
        sidxs = np.argsort(w)[::-1]
        indices = np.argsort(sidxs)
        return _v[indices] , _w[indices]

    def track_modes(self,v,_v,w,_w):
        if w is not None and self.allow_swaps:
            _v[:] , _w[:] = self.swap_modes(w,_w,_v)        
        for gr in self.degen_groups:
            self.correct_degeneracy(gr,v,_v)
        self.make_sign_consistent(v,_v)

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
        _v[group,:] = _vq[:,:].T
        return v,_v,q

    def avg_degen_neff(self,group,neffs):
        neffs[group] = np.mean(neffs[group])[None] 
        return neffs

    def compute_transfer_matrix(self,channel_basis=True,zi=None,zf=None):
        """ compute the transfer matrix M corresponding to propagation through
        the waveguide. propagation of a mode vector v is equivalent to Mv.

        ARGS:
            channel_basis (bool): set True to force the output basis of this matrix to 
            be the waveguide channel basis.
            zi (float or None): initial z value of propagation; if None, use self.zs[0]
            zf (float or None): ifnal z value of propagation; if None, use self.zs[-1] 
        RETURNS:
            (array): an Nmax x Nmax complex-valued transfer matrix.
        """
        N = self.Nmax
        mat = np.zeros((N,N),dtype=np.complex128)
        u0 = np.zeros(N)
        for j in range(N):
            print("\rpropagating mode {0}".format(j),end='',flush=True)
            if j in self.skipped_modes:
                continue
            u0[:] = 0.
            u0[j] = 1.
            zs,us,uf = self.propagate(u0,zi,zf)
            if channel_basis:
                out = self.to_channel_basis(uf)
            else:
                out = uf
            M = len(out)
            mat[:M,j] = out
        return mat

    #endregion
        
    #region mesh gen

    def generate_mesh(self,writeto=None):
        """ generate a mesh for the loaded waveguide according to class attributes.
        
        ARGS:
            writeto (str or None): set to a str (the filename) to write the mesh to file. 
        """
        return self.wvg.make_mesh(writeto=writeto)

    def make_mesh_at_z(self,z):
        """ make the mesh corresponding to the waveguide cross-section at z. 
        
        ARGS:
            z: z coordinate for the mesh.
        
        RETURNS:
            (meshio mesh): the corresponding mesh at z.
        """
        mesh = self.generate_mesh() if self.mesh is None else self.mesh
        if z == 0:
            return copy.deepcopy(mesh)
        return self.wvg.transform_mesh(mesh,0,z)

    #endregion

    #region plotting

    def plot_wavefront(self,zs,us,zi=0,fig=None,ax=None):
        """ plot the complex-valued wavefront through the waveguide. there
        may be some graphical glitches. mesh lines will be visible (issue with 
        matplotlib's tripcolor).
        
        ARGS:
            zs: array of z values
            us: array of mode amplitudes corresponding to the field at each z value
            zi: initial z value for the plot
            fig: a matplotlib figure; if None, one will be made
            ax: a matplotlib axis; if None, one will be made
        
        RETURNS:
            (matplotlib.widgets.slider): the slider object. a reference to the slider needs to be kept to prevent garbage collection from removing it.
        """
        
        plot=False
        if ax is None or fig is None:
            fig,ax = plt.subplots(1,1)
            fig.subplots_adjust(bottom=0.25)
            plot=True

        mesh = copy.deepcopy(self.mesh)
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        x0,y0,w,h = ax.get_position().bounds
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        def update(z):
            ax.clear()
            ix = bisect_left(zs,z)
            f = self.make_field(us[ix],z)
            self.wvg.transform_mesh(self.mesh,0,z,mesh)
            x = mesh.points[:,0]
            y = mesh.points[:,1]
            triangulation = Triangulation(x,y,self.mesh.cells[1].data[:,:3])
            alphas = np.abs(f)
            alphas /= np.max(alphas)
            im = ax.tripcolor(triangulation,np.angle(f),cmap='hsv',vmin=-np.pi,vmax=np.pi,shading="gouraud",alpha=alphas)
            fig.canvas.draw_idle()
            return im,
        slider = None
        im, = update(0)     
        if len(zs) > 1:
            axsl = fig.add_axes([x0,y0-0.25,w,0.1])
            slider = Slider(ax=axsl,label=r'$z$',valmin=zs[0],valmax=zs[-1],valinit=zs[0])
            slider.on_changed(update)
            if zi != 0:
                slider.set_val(zi)   
        if fig is not None:
            fig.colorbar(im,cmap='hsv',ax=ax,ticks=np.linspace(-np.pi,np.pi,5,endpoint=True),label="phase")
        if plot:
            plt.show()
        return slider

    def plot_neffs(self):
        """ plot the effective indices of the eigenmodes """
        neffs = self.neffs.T
        for i in range(self.Nmax):
            plt.plot(self.zs,neffs[i],label="mode "+str(i),)
        for z in self.zs: # plot vertical bars at every z value.
            plt.axvline(x=z,alpha=0.05,color='k',zorder=-100)
        plt.xlabel(r"$z$")
        plt.ylabel("effective index")
        plt.legend(loc='best',bbox_to_anchor=(1.04, 1.))
        plt.tight_layout()
        plt.show()
    
    def plot_neff_diffs(self,yscale="log"):
        """ plot the difference between the effective index of each mode and that of the fundamental.

        ARGS:
            yscale (str): either "lin" for linear scale or "log" for logarithmic scale. this affects only the
            y axis.

        """
        assert yscale in ["log","lin"], "yscale not recognized"
        neffs = self.neffs.T
        for i in range(1,self.Nmax):
            if yscale == "log":
                plt.semilogy(self.zs,neffs[0]-neffs[i],label="mode "+str(i))
            else:
                plt.plot(self.zs,neffs[0]-neffs[i],label="mode "+str(i))
        for z in self.zs: # plot vertical bars at every z value.
            plt.axvline(x=z,alpha=0.05,color='k',zorder=-100)
        plt.xlabel(r"$z$")
        plt.ylabel("difference in index from mode 0")
        plt.legend(loc='best',bbox_to_anchor=(1.04, 1.))
        plt.tight_layout()
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

    def plot_coupling_coeffs(self,legend=True):
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
            ax.axvline(x=z,alpha=0.05,color='k',zorder=-100)
        if legend:
            ax.legend(bbox_to_anchor=(1.04, 1.))
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
        for i in range(us.shape[1]):
            plt.plot(zs,np.power(np.abs(us[:,i]),2),label="mode "+str(i))     
        plt.xlabel(r'$z$ (um)')
        plt.ylabel("power")
        plt.legend(loc='best',bbox_to_anchor=(1.04, 1))
        plt.tight_layout()   
        plt.show()   
            
    def plot_cfield(self,field,z=None,mesh=None,fig=None,ax=None,show_mesh=False,res=1.,xlim=None,ylim=None):
        """ plot a complex-valued field evaluated a finite element mesh. this function is a little
        slow because it resamples <field> onto a grid.
        
        ARGS:
            field: the complex field to be plotted
            z: the z value along the waveguide corresponding to ``field``. alternatively, explicitly pass the next arg
            mesh: the finite element mesh corresponding to ``field``
            fig: matplotlib figure object to receive the plot; if None, one will be made
            ax: matplotlib figure axis object; if None, one will be made
            show_mesh: shows the edges of the FE mesh in the plot
            res: the field will be evaluated on a grid with side length ``res``.
            xlim (tuple): (xmin,xmax) values for the plot - useful if you want to zoom in on the field
            ylim (tuple): (ymin,ymax) values for the plot.
        """
        zinv = (self.zs is not None and len(self.zs)==1) or (self.wvg is not None and self.wvg.z_invariant)
        assert z is not None or mesh is not None or zinv, "one of `z` or `mesh` needs to be passed for plotting"
        if zinv:
            mesh = self.make_mesh_at_z(0) if mesh is None else mesh
        else:
            mesh = self.make_mesh_at_z(z) if mesh is None else mesh
        plot_cfield(field,mesh,fig,ax,show_mesh,res,xlim,ylim)

    def plot_waveguide_mode(self,i,zi=0,fig=None,ax=None):
        """ plot a real-valued eigenmode of the waveguide, from modes saved in self.vs.
        this plot comes with a slider which controls the z value.
        
        ARGS:
            i: the index of the mode to be plotted (mode 0,1,2...)
            zi: starting z value for the plot
            fig: matplotlib figure object; if None, one will be made
            ax: matplotlib axis objects; if None, one will be made
        
        RETURNS:
            (matplotlib.widgets.slider): the slider object. a reference to the slider needs to be kept to prevent garbage collection from removing it.
        """
        plot=False
        if ax is None or fig is None:
            fig,ax = plt.subplots(1,1)
            fig.subplots_adjust(bottom=0.25)
            plot=True

        mesh = copy.deepcopy(self.mesh)
        ax.set_aspect('equal')
        x0,y0,w,h = ax.get_position().bounds
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        def update(z):
            ax.clear()
            v = self.get_v(z)[i]
            self.wvg.transform_mesh(self.mesh,0,z,mesh)
            x = mesh.points[:,0]
            y = mesh.points[:,1]
            triangulation = Triangulation(x,y,self.mesh.cells[1].data[:,:3])
            ax.tripcolor(triangulation,v,shading='gouraud')
            fig.canvas.draw_idle()
        slider = None
        update(0)
        if len(self.zs) > 1:
            axsl = fig.add_axes([x0,y0-0.25,w,0.1])
            slider = Slider(ax=axsl,label=r'$z$',valmin=self.zs[0],valmax=self.zs[-1],valinit=self.zs[0])
            slider.on_changed(update)        
            if zi != 0:
                slider.set_val(zi)
        if plot:
            plt.show()
        return slider # a reference to the slider needs to be preserved to prevent gc :/
    #endregion

class ChainPropagator(Propagator):
    """ a ChainPropagator is a series of Propagators connected `end-to-end`. """

    def __init__(self,propagators:list[Propagator]):
        self.propagators = propagators
        self.z_breaks = [propagators[0].zs[0]]
        for p in propagators:
            self.z_breaks.append(p.zs[-1])

        p0 = propagators[0]
        self.wl = p0.wl
        self.wvg = p0.wvg
        self.Nmax = p0.Nmax
        self.skipped_modes = p0.skipped_modes
        self.mesh = p0.mesh
        self.zs = np.concatenate([p.zs for p in propagators])

    def get_v(self,z):
        return self.get_prop(z).get_v(z)

    def get_prop(self,z):
        idx = max(0,bisect_left(self.z_breaks,z)-1)
        return self.propagators[idx]  

    def propagate(self,u0,zi=None,zf=None):
        if zi is None:
            zi = self.propagators[0].zs[0]
        if zf is None:
            zf = self.propagators[-1].zs[-1]
        
        u = np.array(u0)

        all_zs = None
        all_us = None

        z = zi

        while z != zf:
            p = self.get_prop(z+1e-6) # a little cheap lol
            if zf >= p.zs[-1]:
                zs,us,u = p.propagate(u,z,None)
            else:
                zs,us,u = p.propagate(u,z,zf)
            z = zs[-1]
            
            if all_zs is None:
                all_zs = zs
                all_us = us
            else:
                all_zs = np.concatenate((all_zs,zs))
                all_us = np.concatenate((all_us,us))
        
        return all_zs,all_us,u

    def to_channel_basis(self, uf, z=None):
        if z is None:
            z = self.propagators[-1].zs[-1]
        return self.get_prop(z).to_channel_basis(uf, z)

    def make_field(self, mode_amps, z, plot=False, apply_phase=True):
        return self.get_prop(z).make_field(mode_amps, z, plot, apply_phase)
    