import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio
import copy
from wavesolve.mesher import plot_mesh


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
    def __init__(self,n,points=[]):
        self.points = points
        self.n = n
        self.res = len(points)
        self.mesh_size = None # set to a numeric value to force a triangle size within the closed region
        self.skip_refinement = False
    
    def make_poly(self,geom):
        # check depth of self.points
        if hasattr(self.points[0][0],'__len__'):
            ps = [geom.add_polygon(p) for p in self.points]
            poly = geom.boolean_union(ps)[0]
        else:
            poly = geom.add_polygon(self.points)
        return poly

    def update(self,points):
        """ update the primitive according to some args and return an Nx2 array of points.
            the default behavior is to manually pass in a points array. more specific primitives
            inheriting from Prim2D should implement their own update().
        """
        self.points = points
        self.res = len(self.points)
        return points

    def make_points(self):
        """ make an Nx2 array of points for marking the primitive boundary,
            according to some args.
        """
        return self.points

    def plot_mesh(self):
        """ a quick check to see what this object looks like. generates a mesh with default parameters. """
        with pygmsh.occ.Geometry() as geom:
            poly = self.make_poly(geom)
            geom.add_physical(poly,"poly")
            m = geom.generate_mesh(2,2,6)
        plot_mesh(m)            

    def boundary_dist(self,x,y):
        """ this function computes the distance between the point (x,y) and the boundary of the primitive. negative distances -> inside the boundary, while positive -> outside
            note that this doesn't need to be exact. the "distance" just needs to be positive outside the boundary, negative inside the boundary, and go to 0 as you approach the boundary.
        """
        pass

class Circle(Prim2D):
    """ a Circle primitive, defined by radius, center, and number of sides """
    
    def make_points(self,radius,res,center=(0,0)):
        thetas = np.linspace(0,2*np.pi,res,endpoint=False)
        points = []
        for t in thetas:
            points.append((radius*np.cos(t)+center[0],radius*np.sin(t)+center[1]))        
        points = np.array(points)

        self.radius = radius # save params for later comp
        self.center = center # 

        return points
    
    def boundary_dist(self, x, y):
        return np.sqrt(np.power(x-self.center[0],2)+np.power(y-self.center[1],2)) - self.radius

class Rectangle(Prim2D):
    """ rectangle primitive, defined by corner pounts. """

    def make_points(self,xmin,xmax,ymin,ymax):
        points = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        self.bounds = [xmin,xmax,ymin,ymax]
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
    def __init__(self,p1:Prim2D,p2:Prim2D):
        assert p1.n == p2.n, "primitives must have the same refractive index"
        super().__init__(p1.n,np.array([p1.points,p2.points]))
        self.p1 = p1
        self.p2 = p2

    def make_points(self,args1,args2):
        points1 = self.p1.make_points(args1)
        points2 = self.p2.make_points(args2)
        points = np.array([points1,points2])
        return points
    
    def boundary_dist(self,x,y): # does this need to be vectorized? idk
        return min(self.p1.boundary_dist(x,y),self.p2.boundary_dist(x,y))
        
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
        points = self.make_points_at_z(z)
        self.prim2D.update(points)

    def make_points_at_z(self,z):
        """ make points of prim2D at given z coord. should be implemented by 
            inheriting classes. """
        return self.prim2D.points

    def make_poly_at_z(self,geom,z):
        self.update(z)
        return self.prim2D.make_poly(geom)
    
    def IOR_diff(self,z,dz):
        def _inner(x,y):
            self.update(z)
            inside1 = self.prim2D.boundary_dist(x,y) <=0
            self.update(z+dz)
            inside2 = self.prim2D.boundary_dist(x,y) <=0

            if inside1 and not inside2:
                return -1
            elif inside2 and not inside1:
                return 1
            else:
                return 0
        return _inner

class Pipe(Prim3D):
    """ a Pipe is a 3D primitive with circular cross section at all z. """
    def __init__(self,n,label,res,rfunc,cfunc):
        """
        ARGS:
            n: the refractive index inside the pipe
            label: a string name to attach to this pipe
            res: the number of line segments used to resolve the circle
            rfunc: function that returns a circular radius for a given z
            cfunc: a function that returns a center position (xc,yc) for a given z
        """
        self.rfunc = rfunc
        self.cfunc = cfunc
        self.res = res
        self.n = n
        _circ= Circle(n)
        super().__init__(_circ,label)
    
    def make_points_at_z(self,z):
        points = self.prim2D.make_points(self.rfunc(z),self.res,self.cfunc(z))
        return points

class Box(Prim3D):
    """ an InfiniteBox is a volume whose cross-section has a constant rectangular shape.
        because the shape does not change, we initialize according to the 'starting' box
        geometry, unlike in Pipe where we initialized with functions.
    """
    def __init__(self,n,label,xmin,xmax,ymin,ymax):
        rect = Rectangle(n)
        points = rect.make_points(xmin,xmax,ymin,ymax)
        rect.update(points)
        super().__init__(rect,label)

class ScalingBox(Prim3D):
    """ an box whose width and height scale with z.
    """
    def __init__(self,n,label,xwfunc,ywfunc):
        rect = Rectangle(n)
        super().__init__(rect,label)
        self.xwfunc = xwfunc
        self.ywfunc = ywfunc
    
    def make_points_at_z(self, z):
        points = self.prim2D.make_points(-self.xwfunc(z)/2,self.xwfunc(z)/2,-self.ywfunc(z)/2,self.ywfunc(z)/2)
        return points

#endregion

#region Waveguide
        
class Waveguide:
    """ a Waveguide is a collection of prim3Dgroups, organized into layers. the refractive index 
    of earler layers is overwritten by later layers.
    """

    isect_skip_layers = [0]

    # mesh params
    mesh_dist_scale = 1.0   # mesh boundary refinement linear distance scaling   
    mesh_dist_power = 1.0   # mesh boundary refinement power scaling
    min_mesh_size = 0.1     # minimum allowed mesh size
    max_mesh_size = 10.     # maximum allowed mesh size

    recon_midpts = False

    def __init__(self,prim3Dgroups):
        self.prim3Dgroups = prim3Dgroups # an arrangement of Prim3D objects, stored as a (potentially nested) list. each element is overwritten by the next.
        self.IOR_dict = {}
        self.update(0) # default behavior: init with z=0 for all primitives
        self.z_ex = None # z extent
        
        primsflat = [] # flat array of primitives
        for i,p in enumerate(self.prim3Dgroups):
            if type(p) == list:    
                for _p in p:
                    primsflat.append(_p.prim2D)
            else:
                primsflat.append(p.prim2D)  
        
        self.primsflat = primsflat

    def update(self,z):
        for p in self.prim3Dgroups:
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
            gmsh.option.setNumber('General.Terminal', 0)
            # make the polygons
            polygons = []
            for el in self.prim3Dgroups:
                if type(el) != list:
                    #polygons.append(geom.add_polygon(el.prim2D.points))
                    polygons.append(el.prim2D.make_poly(geom))
                else:
                    els = []
                    for _el in el:
                        #els.append(geom.add_polygon(_el.prim2D.points))
                        els.append(_el.prim2D.make_poly(geom))
                    polygons.append(els)

            # diff the polygons
            for i in range(0,len(self.prim3Dgroups)-1):
                polys = polygons[i]
                _polys = polygons[i+1]
                polys = geom.boolean_difference(polys,_polys,delete_other=False,delete_first=True)
            for i,el in enumerate(polygons):
                if type(el) == list:
                    # group by labels
                    labels = set([p.label for p in self.prim3Dgroups[i]])
                    for l in labels:
                        gr = []
                        for k,poly in enumerate(el):
                            if self.prim3Dgroups[i][k].label == l:
                                gr.append(poly)
                        geom.add_physical(gr,l)
                else:
                    geom.add_physical(el,self.prim3Dgroups[i].label)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
            return mesh
    
    def compute_mesh_size(self,x,y,_scale=1.,_power=1.,min_size=None,max_size=None):
        """ compute a target mesh size (triangle side length) at the point (x,y). 
        ARGS:
            x: x point to compute mesh size at
            y: y point to compute mesh size at
            _scale: a factor that determines how quickly mesh size should increase away from primitive boundaries. higher = more quickly.
            _power: another factor that determines how mesh size increases away from boundaries. default = 1 (linear increase). higher = more quickly.
            min_size: the minimum mesh size that the algorithm can choose
            max_size: the maximum mesh size that the algorithm can chooose
        """
        
        prims = self.primsflat

        dists = np.zeros(len(prims)) # compute a distance to each primitive boundary
        for i,p in enumerate(prims): 
            if p.skip_refinement and p.mesh_size is not None:
                dists[i] = 0. # if there is a set mesh size and we dont care about boundary refinement, set dist=0 -> fixed mesh size inside primitive later
            else:
                dists[i] = p.boundary_dist(x,y)

        # compute target mesh sizes
        mesh_sizes = np.zeros(len(prims))
        for i,d in enumerate(dists): 
            p = prims[i]
            boundary_mesh_size = np.sqrt((p.points[0,0]-p.points[1,0])**2 + (p.points[0,1]-p.points[1,1])**2) 
            scaled_size = np.power(1+np.abs(d)/boundary_mesh_size *_scale ,_power) * boundary_mesh_size # this goes to boundary_mesh_size as d->0, and increases as d->inf for _power>0
            if d<=0 and p.mesh_size is not None:
                mesh_sizes[i] = min(scaled_size,p.mesh_size)
            else:
                mesh_sizes[i] = scaled_size
        target_size = np.min(mesh_sizes)
        if min_size:
            target_size = max(min_size,target_size)
        if max_size:
            scaled_size = min(max_size,target_size)    
        return target_size
    
    def make_mesh_bndry_ref(self,writeto=None):
        """ construct a mesh with boundary refinement at material interfaces."""

        _scale = self.mesh_dist_scale
        _power = self.mesh_dist_power
        min_mesh_size = self.min_mesh_size
        max_mesh_size = self.max_mesh_size

        algo = 6
        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber('General.Terminal', 0)
            # make the polygons
            polygons = []
            for el in self.prim3Dgroups:
                if type(el) != list:
                    #polygons.append(geom.add_polygon(el.prim2D.points))
                    polygons.append(el.prim2D.make_poly(geom))
                else:
                    els = []
                    for _el in el:
                        #els.append(geom.add_polygon(_el.prim2D.points))
                        els.append(_el.prim2D.make_poly(geom))
                    polygons.append(els)

            # diff the polygons
            for i in range(0,len(self.prim3Dgroups)-1):
                polys = polygons[i]
                _polys = polygons[i+1]
                polys = geom.boolean_difference(polys,_polys,delete_other=False,delete_first=True)

            # add physical groups
            for i,el in enumerate(polygons):
                if type(el) == list:
                    # group by labels
                    labels = set([p.label for p in self.prim3Dgroups[i]])
                    for l in labels:
                        gr = []
                        for k,poly in enumerate(el):
                            if self.prim3Dgroups[i][k].label == l:
                                gr.append(poly)

                        geom.add_physical(gr,l)
                else:
                    geom.add_physical(el,self.prim3Dgroups[i].label)

            # mesh refinement callback
            def callback(dim,tag,x,y,z,lc):
                return self.compute_mesh_size(x,y,_scale=_scale,_power=_power,min_size=min_mesh_size,max_size=max_mesh_size)

            geom.env.removeAllDuplicates()
            geom.set_mesh_size_callback(callback)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
            if writeto is not None:
                gmsh.write(writeto+".msh")
                gmsh.clear()
            return mesh

    def make_intersection_mesh(self,z,dz,writeto=None):
        """ construct a mesh around the union of Waveguide boundaries computed at z and z+dz.
            returns both the mesh and a custom dictionary mapping regions of the mesh to
            refractive indices. to advance the refractive index profile from z to z+dz
            use self.advance_IOR() to update the dictionary.
        """
        _scale = self.mesh_dist_scale
        _power = self.mesh_dist_power
        min_mesh_size = self.min_mesh_size
        max_mesh_size = self.max_mesh_size

        IOR_dict={}

        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber('General.Terminal', 0)
            self.update(z)

            # elements
            elmnts = []

            for i,gr in enumerate(self.prim3Dgroups):
                if type(gr) != list:
                    if i in self.isect_skip_layers:
                        elmnts.append([gr.prim2D.make_poly(geom)])
                    else:
                        poly = gr.prim2D.make_poly(geom)
                        gr.update(z+dz)
                        next_poly = gr.prim2D.make_poly(geom)
                        pieces = boolean_fragment(geom,poly,next_poly)
                        elmnts.append([pieces])
                else:
                    if i in self.isect_skip_layers:
                        all_pieces = [p.prim2D.make_poly(geom) for p in gr]
                    else:
                        all_pieces = []
                        for prim in gr:
                            poly = prim.prim2D.make_poly(geom)
                            prim.update(z+dz)
                            next_poly = prim.prim2D.make_poly(geom)
                            pieces = boolean_fragment(geom,poly,next_poly)
                            all_pieces.append(pieces)
                    elmnts.append(all_pieces)

            #boolean subtraction of layers 
            for i in range(len(elmnts)-1):
                group = elmnts[i]
                _group = elmnts[i+1]
                for j,fragroup in enumerate(group):
                    for _fragroup in _group:
                        idx = 0
                        _idx = 0
                        if type(fragroup) == list and fragroup[0] is None:
                            idx = 1
                        if type(_fragroup) == list and _fragroup[0] is None:
                            _idx = 1

                        if type(fragroup) == list:
                            if type(_fragroup) == list:
                                fragroup[idx:] = geom.boolean_difference(fragroup[idx:],_fragroup[_idx:],delete_first=True,delete_other=False)
                            else:
                                fragroup[idx:] = geom.boolean_difference(fragroup[idx:],_fragroup,delete_first=True,delete_other=False)
                        else:
                            if type(_fragroup) == list:
                                fragroup = geom.boolean_difference(fragroup,_fragroup[_idx:],delete_first=True,delete_other=False)
                            else:
                                fragroup = geom.boolean_difference(fragroup,_fragroup,delete_first=True,delete_other=False)

            #add labels
            prim3Dgroups =[]
            for gr in self.prim3Dgroups:
                if type(gr) == list:
                    prim3Dgroups.append(gr)
                else:
                    prim3Dgroups.append([gr])

            for i,layer in enumerate(elmnts):
                for j,sublayer in enumerate(layer):
                    if type(sublayer) == list:
                        # lists contain two or three fragments, formed by the union of the 3D primitives evaled at z and z + dz
                        if len(sublayer)==2:
                            geom.add_physical(sublayer[0],prim3Dgroups[i][j].label+"2")
                            IOR_dict[prim3Dgroups[i][j].label+"2"] = prim3Dgroups[i][j].prim2D.n
                            geom.add_physical(sublayer[1],prim3Dgroups[i][j].label+"1")
                            if type(self.prim3Dgroups[i-1]) == list:
                                IOR_dict[prim3Dgroups[i][j].label+"1"] = prim3Dgroups[i-1][j].prim2D.n
                            else:
                                IOR_dict[prim3Dgroups[i][j].label+"1"] = prim3Dgroups[i-1][0].prim2D.n

                        elif len(sublayer)==3:
                            if sublayer[0] is not None:
                                geom.add_physical(sublayer[0],self.prim3Dgroups[i][j].label+"2")
                                IOR_dict[prim3Dgroups[i][j].label+"2"] = prim3Dgroups[i][j].prim2D.n

                            geom.add_physical(sublayer[1],self.prim3Dgroups[i][j].label+"0")
                            IOR_dict[prim3Dgroups[i][j].label+"0"] = prim3Dgroups[i][j].prim2D.n

                            geom.add_physical(sublayer[2],self.prim3Dgroups[i][j].label+"1")
                            if type(self.prim3Dgroups[i-1]) == list:
                                IOR_dict[prim3Dgroups[i][j].label+"1"] = prim3Dgroups[i-1][j].prim2D.n
                            else:
                                IOR_dict[prim3Dgroups[i][j].label+"1"] = prim3Dgroups[i-1][0].prim2D.n

                        else:
                            raise Exception("wrong number of fragments produced by geom.boolean_fragments()")
                        
                    else:
                        # if the sublayer is just a polygon, its structure is assumed to be fixed from z -> z + dz
                        if type(self.prim3Dgroups[i]) == list:
                            geom.add_physical(layer,self.prim3Dgroups[i].label)
                            IOR_dict[self.prim3Dgroups[i].label] = self.prim3Dgroups[i].prim2D.n
                        else:
                            geom.add_physical(layer,self.prim3Dgroups[i].label)
                            IOR_dict[self.prim3Dgroups[i].label] = self.prim3Dgroups[i].prim2D.n
                        break
                
            # mesh refinement callback
            def callback(dim,tag,x,y,z,lc):
                return self.compute_mesh_size(x,y,_scale=_scale,_power=_power,min_size=min_mesh_size,max_size=max_mesh_size)

            geom.env.removeAllDuplicates()
            geom.set_mesh_size_callback(callback)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=6)
            if writeto is not None:
                gmsh.write(writeto+".msh")
                gmsh.clear()
            return mesh,IOR_dict

    def assign_IOR(self):
        """ build a dictionary which maps all material labels in the Waveguide mesh
            to the corresponding refractive index value """
        for p in self.prim3Dgroups:
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

    def plot_mesh(self,mesh=None,IOR_dict=None,alpha=0.3,ax=None,plot_points=True):
        """ plot a mesh and associated refractive index distribution
        Args:
        mesh: the mesh to be plotted. if None, we auto-compute a mesh using default values
        IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
        """
        show=False
        verts=3
        if ax is None:
            fig,ax = plt.subplots(figsize=(5,5))
            show=True

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

        ax.set_xlim(np.min(points[:,0]),np.max(points[:,0]) )
        ax.set_ylim(np.min(points[:,1]),np.max(points[:,1]) )
        if plot_points:
            for point in points:
                ax.plot(point[0],point[1],color='0.5',marker='o',ms=1.5,alpha=alpha)

        if show:
            plt.show()
    
    def plot_boundaries(self):
        """ plot the boundaries of all prim3Dgroups. For unioned primitives, all boundaries of 
            the original parts of the union are plotted in a lighter color. """
        for group in self.prim3Dgroups:
            if not type(group) == list:
                group = [group]
            for prim in group:
                p = prim.prim2D.points
                if hasattr(p[0][0],'__len__'):
                    for _p in p:
                        p2 = np.zeros((_p.shape[0]+1,_p.shape[1]))
                        p2[:-1] = _p[:]
                        p2[-1] = _p[0]
                        plt.plot(p2.T[0],p2.T[1],color='0.5')
                else:
                    p2 = np.zeros((p.shape[0]+1,p.shape[1]))
                    p2[:-1] = p[:]
                    p2[-1] = p[0]
                    plt.plot(p2.T[0],p2.T[1],color='k')
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
        if self.recon_midpts:
            self.reconstruct_midpoints(mesh)
        return mesh
    
    def reconstruct_midpoints(self,mesh):
        for el in mesh.cells[1].data:
            mesh.points[el[3]] = 0.5*(mesh.points[el[0]]+mesh.points[el[1]])
            mesh.points[el[4]] = 0.5*(mesh.points[el[1]]+mesh.points[el[2]])
            mesh.points[el[5]] = 0.5*(mesh.points[el[2]]+mesh.points[el[0]])

    def IORsq_diff(self,d):
        _d = copy.copy(d)
        for k in _d.keys():
            _k = k[:-1]
            i = k[-1]
            if i == "0":
                dif = d[_k+"2"]**2 - d[_k+"1"]**2
                _d[k] = -dif
            elif i == "1":
                dif = d[_k+"2"]**2 - d[_k+"1"]**2
                _d[k] = dif
            else:
                _d[k] = 0.
        return _d

    
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

        for k,(c,r,n) in enumerate(zip(core_pos,rcores,ncores)):
            cores.append(Pipe(n,"core"+str(k),core_res,rfunc(r),cfunc(c)))
            cores[k].mesh_size = core_mesh_size

        cladrfunc = lambda z: taper_func(z)*rclad
        cladcfunc = lambda z: (0,0)
        _clad = Pipe(nclad,"cladding",clad_res,cladrfunc,cladcfunc)
        _clad.mesh_size = clad_mesh_size

        jackrfunc = lambda z: taper_func(z)*rjack
        jackcfunc = lambda z: (0,0)
        _jack = Pipe(njack,"jacket",jack_res,jackrfunc,jackcfunc)
        _jack.skip_refinement = True

        els = [_jack,_clad,cores]
        
        super().__init__(els)
        self.z_ex = z_ex

    def transform(self,x0,y0,z0,z):
        scale =  self.taper_func(z)/self.taper_func(z0)
        return x0  * scale , y0  * scale

    def isolate(self,k):
        IOR_dict = copy.copy(self.IOR_dict)
        for i in range(len(self.prim3Dgroups[-1])):
            if i != k:
                IOR_dict["core"+str(i)] = IOR_dict["cladding"]
        return IOR_dict

    def isolate_isect(self,k,d):
        IOR_dict = copy.copy(d)

        for i in range(len(self.prim3Dgroups[-1])):
            if i != k:
                IOR_dict["core"+str(i)+"0"] = IOR_dict["cladding2"]
                IOR_dict["core"+str(i)+"1"] = IOR_dict["cladding2"]
                IOR_dict["core"+str(i)+"2"] = IOR_dict["cladding2"]
        return IOR_dict

class Dicoupler(Waveguide):
    """ generic class for directional couplers made of pipes """

    recon_midpts = True

    def __init__(self,rcore1,rcore2,ncore1,ncore2,dmax,dmin,nclad,coupling_length,a,core_res,core_mesh_size,clad_mesh_size):
        
        z_ex = coupling_length * 2 # roughly the middle half is coupling length

        def c2func(z):
            # waveguide channels will follow the blend (func)
            if z <= z_ex/2:
                b = blend(z,z_ex/4-a/2,a)
                #b = blend(z,z_ex/4,a)
                return  np.array([dmin/2,0])*b + np.array([dmax/2,0])*(1-b)
            else:
                b = blend(z,3*z_ex/4+a/2,a)
                #b = blend(z,3*z_ex/4,a)
                return np.array([dmax/2,0])*b + np.array([dmin/2,0])*(1-b)

        def c1func(z):
            return -c2func(z)
        
        self.c1func = c1func
        self.c2func = c2func

        def dfunc(z):
            """ inter core spacing function """
            return c2func(z)[0]-c1func(z)[0]
        
        self.dfunc = dfunc
        self.eps = 1e-12

        cladding = ScalingBox(nclad,"cladding",lambda z: self.dfunc(z)*6,lambda z: self.dfunc(z)*4)
        cladding.mesh_size = clad_mesh_size
        cladding.skip_refinement = True

        core1 = Pipe(ncore1,"core1",core_res,lambda z: rcore1,c1func)
        core1.mesh_size = core_mesh_size

        core2 = Pipe(ncore2,"core2",core_res,lambda z: rcore2,c2func)
        core2.mesh_size = core_mesh_size

        els = [cladding,[core1,core2]]
        self.rcore1 = rcore1
        self.rcore2 = rcore2
        super().__init__(els)
        self.z_ex = z_ex

    def transform_naive(self,x0,y0,z0,z):
        xscale = (self.dfunc(z)-self.rcore1-self.rcore2-2*self.eps)/(self.dfunc(z0)-self.rcore1-self.rcore2-2*self.eps)
        c1_0 = self.c1func(z0)
        c2_0 = self.c2func(z0)
        dd = (self.dfunc(z)-self.dfunc(z0))/2

        x1 = np.where( np.logical_and(c1_0[0]+self.rcore1+self.eps < x0, x0 < c2_0[0]-self.rcore2-self.eps), x0*xscale, x0 )
        x2 = np.where( x1 <= c1_0[0]+self.rcore1+self.eps , x1-dd, x1)
        x3 = np.where( x2 >= c2_0[0]-self.rcore2-self.eps , x2+dd, x2)
        return x3,y0

    def transform(self,x0,y0,z0,z):
        """ better dicoupler transform: for each point (x0,y0), draw the shortest line connecting to each single-mode core boundary.
        the two intersection points and the point (x0,y0) form a triangle. as the separation between the two dicoupler channels changes, allow this triangle to scale (approximately) uniformly
        to a geometrically similar triangle. the transformed point follows the path of the third triangle vertex.
        """

        c1 = self.c1func(z0)
        c2 = self.c2func(z0)
        
        _c1 = self.c1func(z)
        _c2 = self.c2func(z)

        shift1 = _c1[0]-c1[0]
        shift2 = _c2[0]-c2[0]

        newx = np.zeros_like(x0)
        newy = np.zeros_like(y0)

        for i,(x,y) in enumerate(zip(x0,y0)):
            dr1=self.prim3Dgroups[-1][0].prim2D.boundary_dist(x,y)
            dr2=self.prim3Dgroups[-1][1].prim2D.boundary_dist(x,y)
            if dr1<=0:
                newx[i] = x + shift1
                newy[i] = y
                continue
            elif dr2<=0:
                newx[i] = x + shift2
                newy[i] = y
                continue
            
            x1 = x-c1[0]
            y1 = y-c1[1]
            x2 = x-c2[0]
            y2 = y-c2[1]

            t1 = np.arctan2(y1,x1)
            t2 = np.arctan2(y2,x2)
            xc1,yc1 = self.rcore1*np.cos(t1)+c1[0],self.rcore1*np.sin(t1)+c1[1]
            xc2,yc2 = self.rcore2*np.cos(t2)+c2[0],self.rcore2*np.sin(t2)+c2[1]

            new_xc1 = xc1 + shift1
            new_xc2 = xc2 + shift2

            old_side_length_1 = np.sqrt((xc2-xc1)**2+(yc2-yc1)**2)
            new_side_length_1 = np.sqrt((new_xc2-new_xc1)**2+(yc2-yc1)**2)
            scaling = new_side_length_1/old_side_length_1
            if x<0:
                new_dr1 = dr1*scaling
                nx = (self.rcore1+new_dr1)*np.cos(t1) + _c1[0]
                ny = (self.rcore1+new_dr1)*np.sin(t1) + _c1[1]
                newx[i] = nx
                newy[i] = ny
            else:
                new_dr2 = dr2*scaling
                nx = (self.rcore2+new_dr2)*np.cos(t2) + _c2[0]
                ny = (self.rcore2+new_dr2)*np.sin(t2) + _c2[1]
                newx[i] = nx
                newy[i] = ny       
        return newx,newy


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
            IOR_dict["core2"] = IOR_dict["cladding"]
        else:
            IOR_dict["core1"] = IOR_dict["cladding"]
        return IOR_dict

#endregion