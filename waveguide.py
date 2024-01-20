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
### change of basis stuff: add functions to "isolate" cores, so you can compute "unperturbed" modes (in progress)
### add tricoupler class

#region miscellaneous functions   

def load_meshio_mesh(meshname):
    """ ridiculous that i had to write this """
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

def boolean_fragment(geom:pygmsh.occ.Geometry,object,tool):
    """ fragment the tool and the object, and return the fragments in the following order:
        intersection, object_fragment, tool_fragment.
        in some cases one of the later two may be empty
    """
    object_copy = geom.copy(object)
    tool_copy = geom.copy(tool)
    try:
        intersection = geom.boolean_intersection([object_copy,tool_copy])
    except:
        # no intersection - make first element None to signal
        return [None,object,tool]

    object = geom.boolean_difference(object,intersection,delete_first=True,delete_other=False)
    tool = geom.boolean_difference(tool,intersection,delete_first=True,delete_other=False)
    return intersection+object+tool

def linear_taper(final_scale,z_ex):
    def _inner_(z):
        return (final_scale - 1)/z_ex * z + 1
    return _inner_

def blend(z,zc,a):
    """ this is a function of z that smoothly varies from 0 to 1, used to blend functions together. """
    return 0.5 + 0.5 * np.tanh((z-zc)/a)

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
        self.uniform_interior = False # set true to bypass mesh refinement inside the region (the code will to make triangles unfiormly sized)
    
    def make_poly(self):
        with pygmsh.occ.Geometry() as geom:
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

#endregion    

#region Prim3D
class Prim3D:
    """ a Prim3D (3D primitive) is a function of z that returns a Prim2D. """

    def __init__(self,_Prim2D:Prim2D,label:str):
        self._Prim2D = _Prim2D
        self.label = label

        self._uniform_interior = False
        self._mesh_size = None

    @property
    def uniform_interior(self):
        return self._uniform_interior
    @uniform_interior.setter
    def uniform_interior(self,val):
        self._uniform_interior = val
        self._Prim2D.uniform_interior = val
    @property
    def mesh_size(self):
        return self._mesh_size
    @mesh_size.setter
    def mesh_size(self,val):
        self._mesh_size = val
        self._Prim2D.mesh_size = val

    def update(self,z):
        pass

    def make_poly_at_z(self,z):
        self.update(z)
        return self._Prim2D.make_poly()

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
        points = self._Prim2D.make_points(self.rfunc(z),self.res,self.cfunc(z))
        self._Prim2D.update(points,self.n)

class InfiniteBox(Prim3D):
    """ an InfiniteBox is a volume whose cross-section has a constant rectangular shape. """
    def __init__(self,xmin,xmax,ymin,ymax,n,label):
        rect = Rectangle(xmin,xmax,ymin,ymax,n)
        super()._init__(rect,label)
#endregion

#region Waveguide
        
class Waveguide:
    """ a Waveguide is a collection of Prim3Ds, organized into layers. the refractive index 
    of earler layers is overwritten by later layers.
    """
    skip_layers=[]
    
    def __init__(self,Prim3Dgroups):
        self.Prim3Dgroups = Prim3Dgroups # these are "layers" for the optical structure. each layer overwrites the next.
        self.IOR_dict = {}
        self.update(0) # default behavior: init with z=0 for all primitives

    def update(self,z):
        for group in self.Prim3Dgroups:
            for prim in group:
                prim.update(z)

    def make_mesh(self,algo=6):
        """ construct a finite element mesh for the Waveguide cross-section at the currently set 
            z coordinate, which in turn is set through self.update(z). note that meshes will not 
            vary continuously with z. this can only be guaranteed by manually applying a transformation
            to the mesh points which takes it from z1 -> z2.
        """
        with pygmsh.occ.Geometry() as geom:
            # make the polygons
            polygons = [[geom.add_polygon(p._Prim2D.points) for p in group] for group in self.Prim3Dgroups]

            # diff the polygons
            for i in range(0,len(self.Prim3Dgroups)-1):

                polys = polygons[i]
                _polys = polygons[i+1]

                for _p in _polys:
                    polys = geom.boolean_difference(polys,_p,delete_other=False,delete_first=True)
            
            for i,el in enumerate(polygons):
                geom.add_physical(el,self.Prim3Dgroups[i][0].label)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
            return mesh
    
    def make_mesh_bndry_ref(self,algo=6,min_mesh_size=0.05,max_mesh_size=1.,size_scale_fac=0.25,_power=1,_align=False,writeto=None):
        """ construct a mesh with boundary refinement at material interfaces."""
        with pygmsh.occ.Geometry() as geom:
            prims=[]
            # flat array of all 2D primitives, skipping layers as needed
            for i,group in enumerate(self.Prim3Dgroups):
                if i in self.skip_layers:
                    continue
                for p in group:
                    prims.append(p._Prim2D)

            # make the polygons
            polygons = []
            for i,group in enumerate(self.Prim3Dgroups):
                polygroup = []
                for j,p in enumerate(group):
                    if _align and i==1 and j==0:
                        # split the cladding
                        N = len(p._Prim2D.points)
                        points0 = p._Prim2D.points[:int(N/2)+1]
                        points1 = np.concatenate((p._Prim2D.points[int(N/2):],np.array([p._Prim2D.points[0]])))
                        polygroup.append(geom.add_polygon(points0))
                        polygroup.append(geom.add_polygon(points1))
                    else:
                        polygroup.append(geom.add_polygon(p._Prim2D.points))
                polygons.append(polygroup)

            # diff the polygons
            for i in range(0,len(self.Prim3Dgroups)-1):
                polys = polygons[i]
                _polys = polygons[i+1]

                for _p in _polys:
                    polys = geom.boolean_difference(polys,_p,delete_other=False,delete_first=True)

            for i,el in enumerate(polygons):
                geom.add_physical(el,self.Prim3Dgroups[i][0].label)

            # mesh refinement callback
            def callback(dim,tag,x,y,z,lc):
                dists = np.zeros(len(prims))
                for i,p in enumerate(prims): 
                    if p.uniform_interior:
                        dists[i] = max(0,p.boundary_dist(x,y))
                    else:
                        dists[i] = abs(p.boundary_dist(x,y))
                    
                mesh_sizes = np.array([min_mesh_size if p.mesh_size is None else p.mesh_size for p in prims])
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
        for group in self.Prim3Dgroups:
            for p in group:
                if p.label in self.IOR_dict:
                    continue
                self.IOR_dict[p.label] = p._Prim2D.n
        return self.IOR_dict

    def plot_mesh(self,mesh=None,IOR_dict=None, verts=3,alpha=0.3):
        """ plot a mesh and associated refractive index distribution
        Args:
        mesh: the mesh to be plotted. if None, we auto-compute a mesh using default values
        IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
        """
        
        fig,ax = plt.subplots(figsize=(5,5))

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
        for group in self.Prim3Dgroups:
            for prim in group:
                p = prim._Prim2D.points
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
        self.skip_layers=[0] # skip boundary layer mesh refinement for layer 0, the jacket.
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
            cores[k].uniform_interior = True
            cores[k].mesh_size = core_mesh_size
            k+=1

        cladrfunc = lambda z: taper_func(z)*rclad
        cladcfunc = lambda z: (0,0)
        _clad = Pipe(cladrfunc,nclad,cladcfunc,clad_res,"cladding")
        _clad.uniform_interior = True
        _clad.mesh_size = clad_mesh_size

        jackrfunc = lambda z: taper_func(z)*rjack
        jackcfunc = lambda z: (0,0)
        _jack = Pipe(jackrfunc,njack,jackcfunc,jack_res,"jacket")

        els = [[_jack],[_clad],cores]
        
        super().__init__(els)

    def transform(self,x0,y0,z0,z):
        scale =  self.taper_func(z)/self.taper_func(z0)
        return x0  * scale , y0  * scale

    def isolate(self,k):
        IOR_dict = copy.copy(self.IOR_dict)
        for i in range(len(self.Prim3Dgroups[-1])):
            if i != k:
                IOR_dict["core"+str(i)] = IOR_dict["cladding"]
        return IOR_dict

class Dicoupler(Waveguide):
    """ generic class for directional couplers made of pipes """
    def __init__(self,rcore1,rcore2,ncore1,ncore2,dmax,dmin,nclad,z_ex,a,core_res,core_mesh_size,clad_mesh_size):
        maxr = max(rcore1,rcore2)
        cladding = InfiniteBox(-2*dmax,2*dmax,-4*maxr,4*maxr,nclad,"cladding")

        def cfunc2(z):
            # waveguide channels will follow the blend (func), which is constructed from tanh
            if z <= z_ex/2:
                b = blend(z,z_ex/4,a)
                return  np.array([dmin/2,0])*b + np.array([dmax/2,0])*(1-b)
            else:
                b = s(z,3*z_ex/4,a)
                return np.array([dmax/2,0])*b + np.array([dmin/2,0])*(1-b)

        def cfunc1(z):
            return -cfunc1(z)
        
        self.c1func = c1func
        self.c2func = c2func

        def dfunc(z):
            if z <= z_ex/2:
                b = blend(z,z_ex/4,a)
                return  dmin*b + dmax*(1-b)
            else:
                b = s(z,3*z_ex/4,a)
                return dmax*b + dmin*(1-b)

        self.dfunc = dfunc

        core1 = Pipe(lambda z: rcore1, lambda z: ncore1, cfunc1, lambda z: core_res,"core0")
        core1.uniform_interior = True
        core1.mesh_size = core_mesh_size

        core2 = Pipe(lambda z: rcore2, lambda z: ncore2, cfunc2, lambda z: core_res,"core1")
        core2.uniform_interior = True
        core2.mesh_size = core_mesh_size

        els = [[cladding],[core1,core2]]
        self.rcore1 = rcore1
        self.rcore2 = rcore2
        self.ncore
        super().__init__(els)

    def transform(self,x0,y0,z0,z):
        xscale = (self.dfunc(z)-self.rcore1-self.rcore2)/(self.dfunc(z0)-self.rcore1-self.rcore2)
        c1_0 = self.cfunc1(z0)
        c2_0 = self.cfunc2(z0)
        x = np.where( c1_0[0]+self.rcore < x0 < c2_0[0]-self.rcore , x0*xscale, x0 )
        return x,y0

    def isolate(self,k):
        IOR_dict = copy.copy(self.IOR_dict)
        if k == 0:
            IOR_dict["core1"] = IOR_dict["cladding"]
        else:
            IOR_dict["core0"] = IOR_dict["cladding"]
        return IOR_dict

#endregion