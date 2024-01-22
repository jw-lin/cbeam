import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio
import copy

### to do notes
### restructure to evolve meshes based on transformations? or maybe have two options: a "transform" and a "remesh" mode (in progress)
### require a function that computes boundary distance (negative -> inside) for all primitives (done)
### clean up the uniform_interior stuff - maybe rename to mesh_size. and move stuff from boundary distance func into bndry_ref_mesh func so that the boundary distance func is actually boundary distance. (done)
### change of basis stuff: add functions to "isolate" cores, so you can compute "unperturbed" modes (done, but could be cleaner)
### add dicoupler (done, appears to be working) 
### test dicoupler for V and d in range where the emiprical formla is accurate (mainly checking period)
### add a sharp dicoupler
### try asymmetric dicoupler, and measure the asymmetry
### add tricoupler class (could be done after publishing)
### rework propagate so it auto propagates through to the end, and can use a transformed z array
### chagne adaptive stepping so that it always complete, using the minimum z step in worst case scenario
### add union func

#region miscellaneous functions   

def load_meshio_mesh(meshname):
    mesh = meshio.read(meshname+".msh")
    keys = list(mesh.cell_sets.keys())[:-1] # ignore bounding entities for now
    _dict = {}
    for key in keys:
        _dict[key] = []

    cells1data = []
    for i,c in enumerate(mesh.cells): # these should all be triangle6
        for key in keys:
            if len(mesh.cell_sets[key][i]) != 0:
                triangle_indices = mesh.cell_sets[key][i]
                Ntris = len(triangle_indices)
                totaltris = len(cells1data)
                cons = mesh.cells[i].data[triangle_indices]
                if len(cells1data) != 0:
                    cells1data = np.concatenate([cells1data,cons])
                else:
                    cells1data = cons
                if len(_dict[key]) != 0:
                    _dict[key] = np.concatenate([_dict[key],np.arange(totaltris,totaltris+Ntris)])
                else:
                    _dict[key] = np.arange(totaltris,totaltris+Ntris)
                continue
    
    # this is to match the format made by pygmsh
    for key,val in _dict.items():
        _dict[key] = [None,val,None]

    mesh.cell_sets=_dict
    mesh.cells[1].data = cells1data
    for i in range(len(mesh.cells)):
        if i == 1:
            continue
        mesh.cells[i]=None
    return mesh

def boolean_fragment(geom:pygmsh.occ.Geometry,_object,tool):
    """ fragment the tool and the object, and return the fragments in the following order:
        intersection, object_fragment, tool_fragment.
        in some cases one of the later two may be empty
    """
    object_copy = geom.copy(_object)
    tool_copy = geom.copy(tool)
    try:
        intersection = geom.boolean_intersection([object_copy,tool_copy])
    except:
        # no intersection - make first element None to signal
        return [None,_object,tool]

    _object = geom.boolean_difference(_object,intersection,delete_first=True,delete_other=False)
    tool = geom.boolean_difference(tool,intersection,delete_first=True,delete_other=False)
    return intersection+_object+tool

def linear_taper(final_scale,z_ex):
    def _inner_(z):
        return (final_scale - 1)/z_ex * z + 1
    return _inner_

def blend(z,zc,a):
    """ this is a function of z that continuously varies from 0 to 1, used to blend functions together. """
    return 0.5 + 0.5 * np.tanh((z-zc)/(0.25*a)) # the 0.25 is kinda empirical lol

#endregion    

#region Prim2D
class Prim2D:
    """ a Prim2D (2D primitive) is an an array of N (x,y) points, shape (N,2), that denote a closed curve (so, a polygon). 
        inside the closed curve, the primitive has refractive index n. 
    """
    def __init__(self,points,n):
        self.points = points
        self.n = n
        self.res = points.shape[0]
        self.center = None
        self.mesh_size = None # set to a numeric value to force a triangle size within the closed region
        self.skip_refinement = False
    
    def make_poly(self,geom):
        # check depth of self.points
        if hasattr(self.points[0][0],'__len__'):
            ps = [geom.add_polygon(p) for p in self.points]
            poly = geom.boolean_union(ps)
        else:
            poly = geom.add_polygon(self.points)
        return poly
    
    def make_points(self):
        """ this function will generate the polygon as a function of some parameters """
        pass

    def update(self,points,n):
        self.points = points
        self.n = n

    def boundary_dist(self,x,y):
        """ this function computes the distance between the point (x,y) and the boundary of the primitive. negative distances -> inside the boundary, while positive -> outside
            note that this doesn't need to be exact. the "distance" just needs to be positive outside the boundary, negative inside the boundary, and go to 0 as you approach the boundary.
        """
        pass

class Circle(Prim2D):
    """ a Circle primitive, defined by radius, center, and number of sides """
    
    def __init__(self,radius,res,n,center=(0,0)):
        """
        ARGS:
            radius: circle radius
            res: resolution of the circle (number of boundary points and hence segments)
            center: the (x,y) location of the circle center, defaults to (0,0)
        """
        super().__init__(self.make_points(radius,res,center),n)
        self.radius = radius
        self.center = center

    def make_points(self,radius,res,center=(0,0)):
    
        thetas = np.linspace(0,2*np.pi,res,endpoint=False)
        points = []
        for t in thetas:
            points.append((radius*np.cos(t)+center[0],radius*np.sin(t)+center[1]))
        
        points = np.array(points)
        return points
    
    def boundary_dist(self, x, y):
        return np.sqrt(np.power(x-self.center[0],2)+np.power(y-self.center[1],2)) - self.radius

class Rectangle(Prim2D):
    """ rectangle primitive, defined by corner pounts. """

    def __init__(self,xmin,xmax,ymin,ymax,n):
        super().__init__(self.make_points(xmin,xmax,ymin,ymax),n)
        self.bounds = [xmin,xmax,ymin,ymax]

    def make_points(self,xmin,xmax,ymin,ymax):
        points = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        return points

    def boundary_dist(self, x, y):
        bounds = self.bounds
        xdist = min(abs(bounds[0]-x),abs(bounds[1]-x))
        ydist = min(abs(bounds[2]-y),abs(bounds[3]-y))
        dist = min(xdist,ydist)
        if bounds[0]<=x<=bounds[1] and bounds[2]<=y<=bounds[3]:
            return -dist
        return dist

class Prim2DUnion(Prim2D):
    def __init__(p1:Prim2D,p2:Prim2D):
        assert p1.n == p2.n, "primitives must have the same refractive index"
        super().__init__([p1.points,p2.points],p1.n)
        self.p1 = p1
        self.p2 = p2

    def make_points(self,args1,args2):
        return [self.p1.make_points(*args1),self.p2.make_points(*args2)]
    
    def boundary_dist(self,x,y): # does this need to be vectorized? idk
        return min(p1.boundary_dist(x,y),p2.boundary_dist(x,y))
        
#endregion    

#region Prim3D
class Prim3D:
    """ a Prim3D (3D primitive) is a function of z that returns a Prim2D. """

    def __init__(self,prim2D:Prim2D,label:str):
        self.prim2D = prim2D
        self.label = label

        self._mesh_size = None
        self._skip_refinement = False

    @property
    def mesh_size(self):
        return self._mesh_size
    @mesh_size.setter
    def mesh_size(self,val):
        self._mesh_size = val
        self.prim2D.mesh_size = val
    @property
    def skip_refinement(self):
        return self._skip_refinement
    @skip_refinement.setter
    def skip_refinement(self,val):
        self._skip_refinement = val
        self.prim2D.skip_refinement = val

    def update(self,z):
        pass

    def make_poly_at_z(self,geom,z):
        self.update(z)
        return self.prim2D.make_poly(geom)

class Pipe(Prim3D):
    """ a Pipe is a 3D primitive with circular cross section at all z. """
    def __init__(self,rfunc,n,cfunc,res,label):
        """
        ARGS:
            rfunc: function that returns a circular radius for a given z
            n: the refractive index inside the pipe
            cfunc: a function that returns a center position (xc,yc) for a given z
            res: the number of line segments used to resolve the circle
            label: a string name to attach to this pipe
        """
        _circ= Circle(rfunc(0),res,n,cfunc(0))
        self.rfunc = rfunc
        self.cfunc = cfunc
        self.res = res
        self.n = n
        super().__init__(_circ,label)
    
    def update(self,z):
        points = self.prim2D.make_points(self.rfunc(z),self.res,self.cfunc(z))
        self.prim2D.update(points,self.n)

class InfiniteBox(Prim3D):
    """ an InfiniteBox is a volume whose cross-section has a constant rectangular shape. """
    def __init__(self,xmin,xmax,ymin,ymax,n,label):
        rect = Rectangle(xmin,xmax,ymin,ymax,n)
        super().__init__(rect,label)
#endregion

#region Waveguide
        
class Waveguide:
    """ a Waveguide is a collection of Prim3Ds, organized into layers. the refractive index 
    of earler layers is overwritten by later layers.
    """

    def __init__(self,prim3Ds):
        self.prim3Ds = prim3Ds # an arrangement of Prim3D objects, stored as a (potentially nested) list. each element is overwritten by the next.
        self.IOR_dict = {}
        self.update(0) # default behavior: init with z=0 for all primitives
        self.z_ex = None # z extent

    def update(self,z):
        for p in self.prim3Ds:
            if type(p) == list:
                for _p in p:
                    _p.update(z)
            else:
                p.update(z)

    def make_mesh(self,algo=6):
        """ construct a finite element mesh for the Waveguide cross-section at the currently set 
            z coordinate, which in turn is set through self.update(z). note that meshes will not 
            vary continuously with z. this can only be guaranteed by manually applying a transformation
            to the mesh points which takes it from z1 -> z2.
        """
        with pygmsh.occ.Geometry() as geom:
            # make the polygons
            polygons = []
            for el in self.prim3Ds:
                if type(el) != list:
                    polygons.append(geom.add_polygon(el.prim2D.points))
                else:
                    els = []
                    for _el in el:
                        els.append(geom.add_polygon(_el.prim2D.points))
                    polygons.append(els)

            # diff the polygons
            for i in range(0,len(self.prim3Ds)-1):
                polys = polygons[i]
                _polys = polygons[i+1]
                polys = geom.boolean_difference(polys,_polys,delete_other=False,delete_first=True)

            for i,el in enumerate(polygons):
                if type(el) == list:
                    # group by labels
                    labels = [p.label for p in self.prim3Ds[i]]
                    for l in labels:
                        gr = []
                        for k,poly in enumerate(el):
                            if self.prim3Ds[i][k].label == l:
                                gr.append(poly)
                        geom.add_physical(gr,l)
                else:
                    geom.add_physical(el,self.prim3Ds[i].label)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
            return mesh
    
    def make_mesh_bndry_ref(self,algo=6,min_mesh_size=0.05,max_mesh_size=1.,size_scale_fac=0.25,_power=1,writeto=None):
        """ construct a mesh with boundary refinement at material interfaces."""
        with pygmsh.occ.Geometry() as geom:
            
            prims=[]
            # flat array of all 2D primitives, skip_refinement as needed
            for i,p in enumerate(self.prim3Ds):
                if type(p) == list:    
                    for _p in p:
                        prims.append(_p.prim2D)
                else:
                    prims.append(p.prim2D)       
            # make the polygons
            polygons = []
            for el in self.prim3Ds:
                if type(el) != list:
                    polygons.append(geom.add_polygon(el.prim2D.points))
                else:
                    els = []
                    for _el in el:
                        els.append(geom.add_polygon(_el.prim2D.points))
                    polygons.append(els)

            # diff the polygons
            for i in range(0,len(self.prim3Ds)-1):
                polys = polygons[i]
                _polys = polygons[i+1]
                polys = geom.boolean_difference(polys,_polys,delete_other=False,delete_first=True)

            # add physical groups
            for i,el in enumerate(polygons):
                if type(el) == list:
                    # group by labels
                    labels = set([p.label for p in self.prim3Ds[i]])
                    for l in labels:
                        gr = []
                        for k,poly in enumerate(el):
                            if self.prim3Ds[i][k].label == l:
                                gr.append(poly)

                        geom.add_physical(gr,l)
                else:
                    geom.add_physical(el,self.prim3Ds[i].label)

            # mesh refinement callback
            def callback(dim,tag,x,y,z,lc):
                dists = np.zeros(len(prims))
                for i,p in enumerate(prims): 
                    if p.skip_refinement and p.mesh_size is not None:
                        dists[i] = 0.
                    elif p.mesh_size is None:
                        dists[i] = np.inf
                    else:
                        if p.mesh_size is not None:
                            dists[i] = max(0,p.boundary_dist(x,y))
                        else:
                            dists[i] = abs(p.boundary_dist(x,y))
                    
                mesh_sizes = np.array([max_mesh_size if p.mesh_size is None else p.mesh_size for p in prims])
                _size = np.min(np.power(dists,_power) * size_scale_fac + mesh_sizes)
                return min(_size,max_mesh_size)

            geom.env.removeAllDuplicates()
            geom.set_mesh_size_callback(callback)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
            if writeto is not None:
                gmsh.write(writeto+".msh")
                gmsh.clear()
            return mesh

    def assign_IOR(self):
        """ build a dictionary which maps all material labels in the Waveguide mesh
            to the corresponding refractive index value """
        for p in self.prim3Ds:
            if type(p) == list:
                for _p in p:
                    if _p.label in self.IOR_dict:
                        continue
                    self.IOR_dict[_p.label] = _p.prim2D.n
            else:
                if p.label in self.IOR_dict:
                    continue
                self.IOR_dict[p.label] = p.prim2D.n  
        return self.IOR_dict

    def plot_mesh(self,mesh=None,IOR_dict=None, verts=3,alpha=0.3):
        """ plot a mesh and associated refractive index distribution
        Args:
        mesh: the mesh to be plotted. if None, we auto-compute a mesh using default values
        IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
        """
        
        fig,ax = plt.subplots(figsize=(5,5))
        ax.set_aspect('equal')
        if mesh is None:
            mesh = self.make_mesh()
        if IOR_dict is None:
            IOR_dict = self.assign_IOR()
        points = mesh.points
        els = mesh.cells[1].data
        materials = mesh.cell_sets.keys()

        IORs = [ior[1] for ior in IOR_dict.items()]
        n,n0 = max(IORs) , min(IORs)

        for material in materials:       
            cval = IOR_dict[material]/(n-n0) - n0/(n-n0)
            cm = plt.get_cmap("inferno")
            color = cm(cval)

            _els = els[tuple(mesh.cell_sets[material])][0,:,0,:]
            for i,_el in enumerate(_els):
                t=plt.Polygon(points[_el[:verts]][:,:2], facecolor=color)
                t_edge=plt.Polygon(points[_el[:verts]][:,:2], lw=0.5,color='0.5',alpha=alpha,fill=False)
                ax.add_patch(t)
                ax.add_patch(t_edge)


        for point in points:
            plt.plot(point[0],point[1],color='0.5',marker='o',ms=1.5,alpha=alpha)

        plt.xlim(np.min(points[:,0]),np.max(points[:,0]) )
        plt.ylim(np.min(points[:,1]),np.max(points[:,1]) )
        plt.show()
    
    def plot_boundaries(self):
        for group in self.prim3Ds:
            for prim in group:
                p = prim.prim2D.points
                p2 = np.zeros((p.shape[0]+1,p.shape[1]))
                p2[:-1] = p[:]
                p2[-1] = p[0]
                plt.plot(p2.T[0],p2.T[1])
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.axis('equal')
        plt.show()
    
    @staticmethod
    def make_IOR_dict():
        """ this function should return an IOR dictionary, for mode solving. overwrite in child classes."""
        pass

    def transform(self,x0,y0,z0,z):
        """
        tapered waveguides are modelled via a z-dependent transformation function that maps a point (x0,y0,z0) -> (x,y,z).
        this should be implemented by subclasses, and should be vectorized over x0,y0.
        """ 
        return x0,y0

    def transform_mesh(self,mesh0,z0,z,mesh=None):
        if mesh is None:
            mesh = copy.deepcopy(mesh0)
        xp0 = mesh0.points[:,0]
        yp0 = mesh0.points[:,1]
        xp,yp = self.transform(xp0,yp0,z0,z)
        mesh.points = np.array([xp,yp]).T
        return mesh
    
class PhotonicLantern(Waveguide):
    ''' generic class for photonic lanterns '''
    def __init__(self,core_pos,rcores,rclad,rjack,ncores,nclad,njack,z_ex,taper_factor,core_res,clad_res=64,jack_res=32,core_mesh_size=0.05,clad_mesh_size=0.2):
        ''' ARGS: 
            core_pos: an array of core positions at z=0
            rcores: an array of core radii at z=0
            rclad: the cladding radius at z=0
            rjack: the jacket radius at z=0 (this sets the outer simulation boundary)
            ncores: an array of refractive indices for the cores
            nclad: the cladding refractive index
            njack: the jacket refractive index
            z_ex: the lantern length
            taper_factor: the amount the lantern scales by, going from z=0 -> z=z_ex
            core_res: the number of line segments to resolve each core-cladding boundary with
            clad_res: the number of line segments to resolve the cladding-jacket boundary
            jack_res: the number of line segments to resolve the outer jacket boundary
            core_mesh_size: the target side length for triangles inside a lantern core, away from the boundary
            clad_mesh_size: the target side length for triangles inside the lantern cladding, away from the boundary
        '''

        taper_func = linear_taper(taper_factor,z_ex)
        self.taper_func = taper_func

        cores = []

        def rfunc(r):
            def _inner_(z):
                return taper_func(z)*r
            return _inner_

        def cfunc(c):
            def _inner_(z):
                return taper_func(z)*c[0],taper_func(z)*c[1]
            return _inner_

        i = 0
        k=0
        for c,r,n in zip(core_pos,rcores,ncores):
            cores.append(Pipe(rfunc(r),n,cfunc(c),core_res,"core"+str(k)))
            cores[k].mesh_size = core_mesh_size
            k+=1

        cladrfunc = lambda z: taper_func(z)*rclad
        cladcfunc = lambda z: (0,0)
        _clad = Pipe(cladrfunc,nclad,cladcfunc,clad_res,"cladding")
        _clad.mesh_size = clad_mesh_size

        jackrfunc = lambda z: taper_func(z)*rjack
        jackcfunc = lambda z: (0,0)
        _jack = Pipe(jackrfunc,njack,jackcfunc,jack_res,"jacket")
        _jack.skip_refinement = True

        els = [_jack,_clad,cores]
        
        super().__init__(els)
        self.z_ex = z_ex

    def transform(self,x0,y0,z0,z):
        scale =  self.taper_func(z)/self.taper_func(z0)
        return x0  * scale , y0  * scale

    def isolate(self,k):
        IOR_dict = copy.copy(self.IOR_dict)
        for i in range(len(self.prim3Ds[-1])):
            if i != k:
                IOR_dict["core"+str(i)] = IOR_dict["cladding"]
        return IOR_dict

class Dicoupler(Waveguide):
    """ generic class for directional couplers made of pipes """
    def __init__(self,rcore1,rcore2,ncore1,ncore2,dmax,dmin,nclad,coupling_length,a,core_res,core_mesh_size,clad_mesh_size):
        
        z_ex = coupling_length * 2 # roughly the middle half is coupling length

        def c2func(z):
            # waveguide channels will follow the blend (func)
            if z <= z_ex/2:
                b = blend(z,z_ex/4-a/2,a)
                return  np.array([dmin/2,0])*b + np.array([dmax/2,0])*(1-b)
            else:
                b = blend(z,3*z_ex/4+a/2,a)
                return np.array([dmax/2,0])*b + np.array([dmin/2,0])*(1-b)

        def c1func(z):
            return -c2func(z)
        
        self.c1func = c1func
        self.c2func = c2func

        def dfunc(z):
            if z <= z_ex/2:
                b = blend(z,z_ex/4,a)
                return  dmin*b + dmax*(1-b)
            else:
                b = blend(z,3*z_ex/4,a)
                return dmax*b + dmin*(1-b)

        self.dfunc = dfunc
                
        maxr = max(rcore1,rcore2)
        cladding_left = InfiniteBox(-dmax,-dfunc(0)/2+rcore1,-4*maxr,4*maxr,nclad,"cladding")
        cladding_middle = InfiniteBox(-dfunc(0)/2+rcore1,dfunc(0)/2-rcore2,-4*maxr,4*maxr,nclad,"cladding")
        cladding_right = InfiniteBox(dfunc(0)/2-rcore2,dmax,-4*maxr,4*maxr,nclad,"cladding")
        cladding = [cladding_left,cladding_middle,cladding_right]
        for c in cladding:
            c.mesh_size = clad_mesh_size
            c.skip_refinement = True
        
        core1 = Pipe(lambda z: rcore1, ncore1, c1func, core_res,"core0")
        core1.mesh_size = core_mesh_size

        core2 = Pipe(lambda z: rcore2, ncore2, c2func, core_res,"core1")
        core2.mesh_size = core_mesh_size

        els = [cladding,[core1,core2]]
        self.rcore1 = rcore1
        self.rcore2 = rcore2
        super().__init__(els)
        self.z_ex = z_ex

    def transform(self,x0,y0,z0,z):
        xscale = (self.dfunc(z)-self.rcore1-self.rcore2)/(self.dfunc(z0)-self.rcore1-self.rcore2)
        c1_0 = self.c1func(z0)
        c2_0 = self.c2func(z0)
        dd = (self.dfunc(z)-self.dfunc(z0))/2

        x1 = np.where( np.logical_and(c1_0[0]+self.rcore1 < x0, x0 < c2_0[0]-self.rcore2), x0*xscale, x0 )
        x2 = np.where( x1 <= c1_0[0]+self.rcore1 , x1-dd, x1)
        x3 = np.where( x2 >= c2_0[0]-self.rcore2 , x2+dd, x2)
        return x3,y0
    
    def plot_paths(self):
        zs = np.linspace(0,self.z_ex,400)
        c1s = [self.c1func(z)[0] for z in zs]
        c2s = [self.c2func(z)[0] for z in zs]
        plt.plot(zs,c1s,label="channel 1")
        plt.plot(zs,c2s,label="channel 2")
        plt.xlabel(r"$z$")
        plt.ylabel(r"$x$")
        plt.legend(loc='best',frameon=False)
        plt.show()

    def isolate(self,k):
        IOR_dict = copy.copy(self.IOR_dict)
        if k == 0:
            IOR_dict["core1"] = IOR_dict["cladding"]
        else:
            IOR_dict["core0"] = IOR_dict["cladding"]
        return IOR_dict

#endregion