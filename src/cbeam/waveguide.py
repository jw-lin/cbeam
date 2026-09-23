import pygmsh
import gmsh
import meshio
import copy
from itertools import combinations

import math
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

# waveguide geometry + FE mesh construction is host-only (gmsh/pygmsh are C
# libraries; the size callback is a scalar hot loop).  It must never run on a
# JAX device array, so `xp` here is always plain numpy regardless of
# CBEAM_BACKEND.  The propagation math (propagator.py / backend.py) is what
# switches backends.
xp = np

def hex_ring_positions(rings, core_spacing=1.0, plot=False):
    """
    Returns (x, y) coordinates for a hexagonal (round) tiling of 'rings' layers.
    Index 0 is guaranteed to be the centre (0,0). The rest are ordered by
    ring radius and then by angle.

    Parameters:
        rings : int
            Number of rings (>= 1).
            rings=1 -> centre only (1 point)
            rings=2 -> centre + 1 ring (7 points)
            rings=3 -> centre + 2 rings (19 points)
            rings=4 -> centre + 3 rings (37 points)
            ...
        core_spacing : float, default=1.0
            Centre-to-centre distance between adjacent hexagons.
        plot : bool, default=False
            If True, displays a plot of the hexagonal tiling with each output
            position labelled by its index.

    Returns:
        list of (float, float)
            Coordinates in the plane, scaled by core_spacing. 
            positions[0] is always (0,0).
    """
    if rings < 1:
        raise ValueError("rings must be >= 1")

    R = rings - 1
    raw_positions = []

    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            if max(abs(q), abs(r), abs(q + r)) <= R:
                x = (q + r / 2.0) * core_spacing
                y = (math.sqrt(3) / 2.0) * r * core_spacing
                ring_rad = max(abs(q), abs(r), abs(q + r))
                angle = math.atan2(y, x)
                raw_positions.append((x, y, ring_rad, angle))

    # Sort: centre first, then by ring, then by angle
    raw_positions.sort(key=lambda p: (p[2], p[3]))
    positions = [(x, y) for x, y, _, _ in raw_positions]

    # Optional plotting
    if plot:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')

        hex_radius = core_spacing / math.sqrt(3)

        # --- ROTATED HEXAGONS BY 90° ---
        # Previously orientation = np.pi/2 (pointy-top)
        # Now orientation = 0 (flat-top) – i.e. rotated by 90°
        hex_orientation = 0.0   # flat-top

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(positions)))

        for i, (x, y) in enumerate(positions):
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=hex_orientation,
                facecolor=colors[i],
                edgecolor='black',
                linewidth=1,
                alpha=0.7
            )
            ax.add_patch(hexagon)
            text_color = 'white' if core_spacing > 0.5 else 'black'
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=8, weight='bold', color=text_color)

        margin = 0.6 * core_spacing
        xs, ys = zip(*positions)
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.set_title(f"{len(positions)} outputs ({rings} ring{'s' if rings>1 else ''}), spacing = {core_spacing}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    return positions

def get_19port_positions(core_spacing):
    # NOTE: the core *ordering* here is load-bearing.  Cores are inserted into
    # the gmsh geometry in this order, which fixes the mesh triangulation and
    # therefore the tie-breaking among the (near-)degenerate guided modes of a
    # 19-port lantern.  This reproduces the exact sequence used upstream
    # (jw-lin/cbeam); do not replace it with hex_ring_positions(), whose
    # angle-sorted order gives a physically identical device but a different
    # discretisation and hence different compute_neffs() output.
    pos= [[0,0]]

    for i in range(6):
        xpos = core_spacing*np.cos(i*np.pi/3)
        ypos = core_spacing*np.sin(i*np.pi/3)
        pos.append([xpos,ypos])
    
    startpos = np.array([2*core_spacing,0])
    startang = 2*np.pi/3
    pos.append(startpos)
    for i in range(11):
        if i%2==0 and i!=0:
            startang += np.pi/3
        nextpos = startpos + np.array([core_spacing*np.cos(startang),core_spacing*np.sin(startang)])
        pos.append(nextpos)
        startpos = nextpos

    return np.array(pos)


# ------------------- Plotting/mesh loading functions ------------------- #
def plot_mesh(mesh,IOR_dict=None,alpha=0.3,ax=None,plot_points=True,verbose=True):
    """ plot a mesh and associated refractive index distribution
    Args:
    mesh: the mesh to be plotted. if None, we auto-compute a mesh using default values
    IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
    """
    import matplotlib.pyplot as plt
    show=False
    verts=3
    if ax is None:
        fig,ax = plt.subplots(figsize=(5,5))
        show=True

    ax.set_aspect('equal')
    points = mesh.points
    if verbose:
        print("mesh has", points.shape[0], "points")
    els = mesh.cells[1].data
    materials = mesh.cell_sets.keys()

    if IOR_dict is not None:
        IORs = [ior[1] for ior in IOR_dict.items()]
        nmax, nmin = max(IORs), min(IORs)

    for material in materials:   
        if IOR_dict is not None:    
            cval = IOR_dict[material] / (nmax-nmin) - nmin/(nmax-nmin)
            cm = plt.get_cmap("inferno")
            color = cm(cval)
        else:
            color="None"

        _els = els[tuple(mesh.cell_sets[material])][0,:,0,:]
        for _el in _els:
            t=plt.Polygon(points[_el[:verts]][:,:2], facecolor=color)
            t_edge=plt.Polygon(points[_el[:verts]][:,:2], lw=0.5,color='0.5',alpha=alpha,fill=False)
            ax.add_patch(t)
            ax.add_patch(t_edge)

    ax.set_xlim(xp.min(points[:,0]), xp.max(points[:,0]))
    ax.set_ylim(xp.min(points[:,1]), xp.max(points[:,1]))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    if plot_points:
        for point in points:
            ax.plot(point[0],point[1],color='0.5',marker='o',ms=1.5,alpha=alpha)

    if show:
        plt.show()

def load_meshio_mesh(meshname):
    mesh = meshio.read(meshname+".msh")
    keys = list(mesh.cell_sets.keys())[:-1] # ignore bounding entities
    _dict = {}
    for key in keys:
        _dict[key] = []
    cells1data = []
    for i, c in enumerate(mesh.cells):
        for key in keys:
            if len(mesh.cell_sets[key][i]) != 0:
                triangle_indices = mesh.cell_sets[key][i]
                Ntris = len(triangle_indices)
                totaltris = len(cells1data)
                cons = mesh.cells[i].data[triangle_indices]
                if len(cells1data) != 0:
                    cells1data = xp.concatenate([cells1data, cons])
                else:
                    cells1data = cons
                if len(_dict[key]) != 0:
                    _dict[key] = xp.concatenate([_dict[key], xp.arange(totaltris, totaltris+Ntris)])
                else:
                    _dict[key] = xp.arange(totaltris, totaltris+Ntris)
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
    """Fragment the tool and the object, and return the fragments in the following order:
    intersection, object_fragment, tool_fragment.
    In some cases one of the later two may be empty.
    """
    object_copy = geom.copy(_object)
    tool_copy = geom.copy(tool)
    try:
        intersection = geom.boolean_intersection([object_copy,tool_copy])
    except Exception:
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
    """Function of z that continuously varies from 0 to 1, used to blend functions together."""
    return 0.5 + 0.5 * xp.tanh((z - zc) / (0.25 * a)) # the 0.25 is kinda empirical lol

def dist(p1,p2,axis=None):
    if axis is not None:
        return xp.sqrt(xp.sum(xp.power(p1 - p2, 2), axis=axis))
    return xp.sqrt(xp.sum(xp.power(p1 - p2, 2)))

# ------------------- Utility math functions ------------------- #
def dist(p1, p2, axis=None):
    if axis is not None:
        return xp.sqrt(xp.sum(xp.power(p1 - p2, 2), axis=axis))
    return xp.sqrt(xp.sum(xp.power(p1 - p2, 2)))

def rotate(v,theta):
    if v.ndim == 2:
        return xp.array([xp.cos(theta)*v[:,0] - xp.sin(theta)*v[:,1],
                         xp.sin(theta)*v[:,0] + xp.cos(theta)*v[:,1]]).T
    else:
        return xp.array([xp.cos(theta)*v[0] - xp.sin(theta)*v[1],
                         xp.sin(theta)*v[0] + xp.cos(theta)*v[1]])

def blend(z, zc, a):
    """ this is a function of z that continuously varies from 0 to 1, used to blend functions together. """
    return 0.5 + 0.5 * xp.tanh((z - zc) / (0.25 * a))

#-------------------- Geometric Primitives: Prim2D, Circle, Rectangle, Union ----------------------#
class Prim2D:
    """ a Prim2D (2D primitive) is an an array of N (x,y) points, shape (N,2), that denote a closed curve (so, a polygon). 
        inside the closed curve, the primitive has refractive index n. 
    """
    def __init__(self,n,points=[]):
        self.points = xp.array(points)
        self.n = n
        self.res = len(points)
        self.mesh_size = None   # set to a number to force triangle size in region
        self.skip_refinement = False
    
    def make_poly(self,geom):
        # gmsh/pygmsh is host-only: always hand it plain numpy, never a JAX array.
        # Depth check via ndim (a JAX 0-d Array exposes __len__, so the old
        # ``hasattr(self.points[0][0], '__len__')`` test wrongly fired on the
        # JAX backend and split a single polygon into 1-D rows).
        pts = np.asarray(self.points)
        if pts.ndim == 3:                       # nested polygons -> union (Prim2DUnion)
            ps = [geom.add_polygon(p) for p in pts]
            poly = geom.boolean_union(ps)[0]
        else:
            poly = geom.add_polygon(pts)
        return poly

    def update(self,points):
        """ update the primitive according to some args and return an Nx2 array of points.
            the default behavior is to manually pass in a points array. more specific primitives
            inheriting from Prim2D should implement their own update().
        """
        self.points = xp.array(points)
        self.res = len(self.points)
        return self.points

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
        # User override for specific primitives
        """ this function computes the distance between the point (x,y) and the boundary of the primitive. 
        negative distances -> inside the boundary, while positive -> outside. note that this doesn't need to be exact, 
        the "distance" just needs to be positive outside the boundary, negative inside the boundary, and go to 0 as you approach the boundary.
        it is better if it works on vectorized inputs for x,y.

        ARGS:
            x (float or vector): x coordinate(s)  of point(s) to measure distance w/ boundary.
            y (float or vector): y coordinate(s)  of point(s) to measure distance w/ boundary.
        RETURNS:
            (float or vector): the distance(s) to the boundary from the point(s)
        """
        raise NotImplementedError("Primitive boundary_dist must be implemented in subclasses.")


    def nearest_boundary_point(self,x,y):
        # User override for specific primitives
        """ this function computes the point on the boundary that is closest to a point (x,y). 
        it is better if it works on vectorized inputs for x,y.

        ARGS:
            x (float or vector): x coordinate(s)  of point(s) to measure distance w/ boundary.
            y (float or vector): y coordinate(s)  of point(s) to measure distance w/ boundary.
        RETURNS:
            (tuple): a tuple containing:
                - (float or vector): the x coordinate(s) of the nearest boundary point     
                - (float or vector): the y coordinate(s) of the nearest boundary point        
        """
        raise NotImplementedError("Primitive nearest_boundary_point must be implemented in subclasses.")

class Circle(Prim2D):
    """ a Circle primitive, defined by radius, center, and number of sides
        (so actually a regular polygon lol). initialize using only 
        the desired refractive index n, then use make_points() to generate
        the point array.
    """
    def make_points(self,radius,res,center=(0,0)):
        thetas = xp.linspace(0, 2*xp.pi, res, endpoint=False)
        points = [ (radius*xp.cos(t)+center[0], radius*xp.sin(t)+center[1]) for t in thetas ]
        points = xp.array(points)
        self.radius = radius
        self.center = center
        return points
    
    def boundary_dist(self, x, y):
        return xp.sqrt(xp.power(x-self.center[0],2)+xp.power(y-self.center[1],2)) - self.radius
    
    def nearest_boundary_point(self, x, y):
        t = xp.arctan2(y-self.center[1], x-self.center[0])
        bx = self.radius*xp.cos(t)
        by = self.radius*xp.sin(t)
        return bx+self.center[0],by+self.center[1]

class Rectangle(Prim2D):
    """ rectangle primitive, defined by corner points. initialize using only 
        the desired refractive index n, then use make_points() to generate
        the point array. """
    def make_points(self,xmin,xmax,ymin,ymax):
        points = xp.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        self.bounds = [xmin,xmax,ymin,ymax]
        return points

    def boundary_dist(self, x, y):
        if type(x) != xp.ndarray:
            x = xp.array([x])
            y = xp.array([y])
        bounds = self.bounds
        xdist = xp.minimum(xp.abs(bounds[0]-x),xp.abs(bounds[1]-x))
        ydist = xp.minimum(xp.abs(bounds[2]-y),xp.abs(bounds[3]-y))
        distv = xp.minimum(xdist,ydist)
        nx,ny = self.nearest_boundary_point(x,y)
        outdist = xp.sqrt(xp.power(nx-x,2)+xp.power(ny-y,2))
        out = xp.where((bounds[0]<=x) & (x<=bounds[1]) & (bounds[2]<=y) & (y<=bounds[3]), -distv, outdist)
        if out.shape[0] == 1:
            return out[0]
        return out
    
    def nearest_boundary_point(self, x, y):
        if type(x) != xp.ndarray:
            x = xp.array([x])
            y = xp.array([y])
        bounds = self.bounds
        xd0,xd1 = xp.abs(bounds[0]-x),xp.abs(bounds[1]-x)
        yd0,yd1 = xp.abs(bounds[2]-y),xp.abs(bounds[3]-y)
        i = xp.argmin(xp.array([xd0,xd1,yd0,yd1]),axis=0)
        outx = xp.zeros_like(x)
        outy = xp.zeros_like(y)
        cond0 = (i==0) & (bounds[2]<=y) & (y<=bounds[3])
        outx[cond0] = bounds[0]
        outy[cond0] = y[cond0]
        cond1 = (i==0) & (y<bounds[2])
        outx[cond1] = bounds[0]
        outy[cond1] = bounds[2]
        cond2 = (i==0) & (y>bounds[3])
        outx[cond2] = bounds[0]
        outy[cond2] = bounds[3]

        cond3 = (i==1) & (bounds[2]<=y) & (y<=bounds[3])
        outx[cond3] = bounds[1]
        outy[cond3] = y[cond3]
        cond4 = (i==1) & (y<bounds[2])
        outx[cond4] = bounds[1]
        outy[cond4] = bounds[2]
        cond5 = (i==1) & (y>bounds[3])
        outx[cond5] = bounds[1]
        outy[cond5] = bounds[3]

        cond6 = (i==2) & (bounds[0]<=x) & (x<=bounds[1])
        outx[cond6] = x[cond6]
        outy[cond6] = bounds[2]
        cond7 = (i==2) & (x<bounds[0])
        outx[cond7] = bounds[0]
        outy[cond7] = bounds[2]
        cond8 = (i==2) & (x>bounds[1])
        outx[cond8] = bounds[1]
        outy[cond8] = bounds[2]

        cond9 = (i==3) & (bounds[0]<=x) & (x<=bounds[1])
        outx[cond9] = x[cond9]
        outy[cond9] = bounds[3]
        cond10 = (i==3) & (x<bounds[0])
        outx[cond10] = bounds[0]
        outy[cond10] = bounds[3]
        cond11 = (i==3) & (x>bounds[1])
        outx[cond11] = bounds[1]
        outy[cond11] = bounds[3]

        if outx.shape[0] == 1:
            return outx[0],outy[0]
        return outx, outy
    
class Prim2DUnion(Prim2D):
    def __init__(self,p1:Prim2D,p2:Prim2D):
        """ initialize a boolean union of two primitives, p1 and p2. 
        not fully tested.
        """
        assert p1.n == p2.n, "primitives must have the same refractive index"
        super().__init__(p1.n, xp.array([p1.points,p2.points]))
        self.p1 = p1
        self.p2 = p2

    def make_points(self,args1,args2):
        """ make points corresponding to the boundary of the primitive. 

        ARGS:
            args1 (tuple): the arguments of make_points for the first primitive
            args2 (tuple): the arguments of make_points for the second primitive
        """
        points1 = self.p1.make_points(*args1)
        points2 = self.p2.make_points(*args2)
        points = xp.array([points1,points2])
        return points
    
    def boundary_dist(self,x,y):
        return min(self.p1.boundary_dist(x,y),self.p2.boundary_dist(x,y))

    def nearest_boundary_point(self, x, y):
        if self.p1.boundary_dist(x,y) < self.p2.boundary_dist(x,y):
            return self.p1.nearest_boundary_point(x,y)
        return self.p2.nearest_boundary_point(x,y)

#---------------------- 3D Primitives: Prim3D, Pipe, BoxPipe, Box etc. -------------------------#
class Prim3D:
    """ a Prim3D (3D primitive) is a function of z that returns a Prim2D. 
    it can be used to represent tapering pipes, boxes, etc.
    """
    preserve_shape = True

    def __init__(self,prim2D:Prim2D,label:str):
        """ initialize a Prim3D object. this default behavior is often
        overwritten by inheriting classes.

        ARGS:
            prim2D (Prim2D): a prim2D object (e.g. Circle, Rectangle).
            label (str): a string label to attach to the 2D primitive.
        """
        self.prim2D = prim2D
        self.label = label
        self._mesh_size = None
        self._skip_refinement = False
        self.n = prim2D.n

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
        """ update self.prim2D to the desired z coordinate. """
        points = self.make_points_at_z(z)
        self.prim2D.update(points)

    def make_points_at_z(self,z):
        """ make points of prim2D at given z coord. should be implemented by 
        inheriting classes. 

        ARGS:
            z (float): the desired z coordinate 

        RETURNS:
            (array): an Nx2 array of points corresponding to the boundary at z.
        """
        return self.prim2D.points

    def transform_point_inside(self,x0,y0,z0,z):
        # By default: no transform.
        """ for a point (x0,y0) inside the boundary at z0, compute a new point (x,y) at z, accounting
        for the z-variation of the Prim3D. should be implemented by inheriting classes, and 
        should be vectorized if possible.

        ARGS:
            x0 (float or vector): reference x coordinate(s) inside the boundary at z0
            y0 (float or vector): reference y coordinate(s) inside the boundary at z0
            z0 (float): reference z coordiante
            z (float): new z coordinate
        RETURNS:
            (tuple): a tuple containing:
                - x1 (float or vector): the new x coordinate(s) at z
                - y1 (float or vector): the new y coordinate(s) at z
        """
        return x0,y0

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
    """
    A Pipe is a 3D primitive with circular cross section at all z.
    """
    def __init__(self,n,label,res,rfunc,cfunc=(0,0)):
        """
        ARGS:
            n: the refractive index inside the pipe
            label: a string name to attach to this pipe
            res: the number of line segments used to resolve the circle
            rfunc: function that returns a circular radius for a given z, or scalar for constant behavior
            cfunc: a function that returns a center position (xc,yc) for a given z, or tuple for constant behavior
        """
        self.rfunc = rfunc if callable(rfunc) else lambda z: rfunc
        self.cfunc = cfunc if callable(cfunc) else lambda z: cfunc
        self.res = res
        self.n = n
        _circ= Circle(n)
        super().__init__(_circ,label)
    
    def make_points_at_z(self,z):
        points = self.prim2D.make_points(self.rfunc(z),self.res,self.cfunc(z))
        return points

    def transform_point_inside(self, x0, y0, z0, z):
        c = self.cfunc(z0)
        _c = self.cfunc(z)
        rscale = self.rfunc(z)/self.rfunc(z0)
        t = xp.arctan2(y0-c[1], x0-c[0])
        r = xp.sqrt(xp.power(x0-c[0],2) + xp.power(y0-c[1],2))
        return r*rscale*xp.cos(t) + _c[0], r*rscale*xp.sin(t) + _c[1]

class LinearPipe(Pipe):
    """
    A LinearPipe is a Pipe whose radius and centerpoint varies linearly.
    """
    def __init__(self,n,label,res,r1,r2,z_ex,c1=(0,0),c2=(0,0)):
        """
        ARGS:
            n: the refractive index inside the pipe
            label: a string name to attach to this pipe
            res: the number of line segments used to resolve the circle
            r1: the starting radius
            r2: the ending radius
            z_ex: the z distance between the start and end of the Pipe
            c1: the starting centerpoint
            c2: the ending centerpoint
        """
        rfunc = lambda z: r1 + z/z_ex * (r2-r1)
        cfunc = lambda z: (c1[0] + z/z_ex * (c2[0]-c1[0]) , c1[1] + z/z_ex * (c2[1]-c1[1]))
        super().__init__(n,label,res,rfunc,cfunc)

class Box(Prim3D):
    """ an Box is a volume whose cross-section has a constant rectangular shape.
        because the shape does not change, we initialize according to the 'starting' box
        geometry, unlike in Pipe where we initialized with functions.
    """
    def __init__(self,n,label,xmin,xmax,ymin,ymax):
        rect = Rectangle(n)
        points = rect.make_points(xmin,xmax,ymin,ymax)
        rect.update(points)
        super().__init__(rect,label)

class BoxPipe(Prim3D):
    """
    An box whose width, height, and center can scale with z.
    """
    def __init__(self,n,label,xwfunc,ywfunc,cfunc=(0,0)):
        """ 
        ARGS:
            n : refractive index
            label : string identifier for this waveguide part
            xwfunc : a function controlling the width of the box wrt z, or scalar for constant x width
            ywfunc : a function controlling the height of the box wrt z, or scalar for constant y width
            cfunc : a function controlling the center of the box, or a tuple for constant center location (x0,y0)
        """
        rect = Rectangle(n)
        super().__init__(rect,label)
        self.xwfunc = xwfunc if callable(xwfunc) else lambda z: xwfunc
        self.ywfunc = ywfunc if callable(ywfunc) else lambda z: ywfunc
        self.cfunc = cfunc if callable(cfunc) else lambda z: cfunc
    
    def make_points_at_z(self, z):
        cx,cy = self.cfunc(z)
        points = self.prim2D.make_points(-self.xwfunc(z)/2+cx,self.xwfunc(z)/2+cx,-self.ywfunc(z)/2+cy,self.ywfunc(z)/2+cy)
        return points

    def transform_point_inside(self, x0, y0, z0, z):
        c = self.cfunc(z0)
        _c = self.cfunc(z)
        xscale = self.xwfunc(z)/self.xwfunc(z0)
        yscale = self.ywfunc(z)/self.ywfunc(z0)
        return (x0-c[0])*xscale + _c[0] ,  (y0-c[1])*yscale + _c[1] 


#------------------------- Waveguide core and practical subclasses --------------------------#
class Waveguide:
    """ a Waveguide is a collection of prim3D objects (primitive 3D shapes like
    circular pipes and boxes), organized into layers as a nested list. 
    the refractive index of earlier layers is overwritten by later layers.
    """
    isect_skip_layers = [0]
    mesh_dist_scale = 0.5
    mesh_dist_power = 1.0 
    min_mesh_size = 0.01
    max_mesh_size = 100.
    linear = False
    recon_midpts = True
    vectorized_transform = True
    fd_eps = 1e-6
    z_invariant = False
    z_ex = None

    def __init__(self,prim3Dgroups):
        """ initialize a Waveguide object.
        
        ARGS:
            prim3Dgroups: a (potentially) nested list of prim3D objects. later
                          top-level elements overwrite earlier ones, in terms
                          of refractive index.
        """
        self.xp = xp  # always numpy here (see module-level note)
        self.prim3Dgroups = prim3Dgroups
        self.IOR_dict = {}
        self.update(0)
        
        primsflat = []
        for i,p in enumerate(self.prim3Dgroups):
            if type(p) == list:    
                for _p in p:
                    primsflat.append(_p.prim2D)
            else:
                primsflat.append(p.prim2D)  
        self.primsflat = primsflat

        prim3Dsflat = []
        for i,p in enumerate(self.prim3Dgroups):
            if type(p) == list:    
                for _p in p:
                    prim3Dsflat.append(_p)
            else:
                prim3Dsflat.append(p)  
        self.prim3Dsflat = prim3Dsflat

    def update(self,z):
        """ update the mesh boundaries to the given z coordinate. 
        
        ARGS:
            z (float):  desired z coordinate. note that all waveguides are 
                        updated to z=0 upon initialization.
        """
        for p in self.prim3Dgroups:
            if type(p) == list:
                for _p in p:
                    _p.update(z)
            else:
                p.update(z)
    def _make_mesh_dep(self,algo=6):
        """
        Deprecated meshing function. Construct a finite element mesh for the Waveguide cross-section at the currently set 
        z coordinate, which in turn is set through self.update(z).
        Note that meshes will not vary continuously with z. This can only be guaranteed by
        manually applying a transformation to the mesh points which takes it from z1 -> z2.
        """
        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber('General.Terminal', 0)
            # make the polygons
            polygons = []
            for el in self.prim3Dgroups:
                if type(el) != list:
                    polygons.append(el.prim2D.make_poly(geom))
                else:
                    els = []
                    for _el in el:
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
                    labels = set(p.label for p in self.prim3Dgroups[i])
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
    
    def _compute_mesh_size(self,x,y,_scale=1.,_power=1.,min_size=None,max_size=None):
        # Hot gmsh size callback: pure host scalar math.  Must stay on plain
        # numpy -- self.xp.zeros() under the JAX backend is immutable and the
        # ``dists[i] = ...`` writes below would raise.
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
        dists = np.zeros(len(prims))
        for i,p in enumerate(prims): 
            if p.skip_refinement and p.mesh_size is not None:
                dists[i] = 0.
            else:
                dists[i] = float(p.boundary_dist(x, y))

        mesh_sizes = np.zeros(len(prims))
        for i,d in enumerate(dists): 
            p = prims[i]
            ms = np.inf if p.mesh_size is None else p.mesh_size
            p0x, p0y = float(p.points[0, 0]), float(p.points[0, 1])
            p1x, p1y = float(p.points[1, 0]), float(p.points[1, 1])
            boundary_mesh_size = min(ms, np.sqrt((p0x - p1x)**2 + (p0y - p1y)**2))
            scaled_size = np.power(1 + np.abs(d)/boundary_mesh_size * _scale, _power) * boundary_mesh_size
            if d<=0 and p.mesh_size is not None:
                mesh_sizes[i] = min(scaled_size,p.mesh_size)
            else:
                mesh_sizes[i] = scaled_size
        target_size = np.min(mesh_sizes)
        if min_size:
            target_size = max(min_size,target_size)
        if max_size:
            # NOTE: upstream (jw-lin/cbeam) assigns to `scaled_size` here, so the
            # max_size clamp is a no-op and `target_size` is returned unclamped.
            # Kept identical for bit-for-bit realignment; with the default
            # max_mesh_size=100 the clamp never binds for the bundled examples
            # anyway (the cladding term caps target_size well below 100).
            scaled_size = min(max_size,target_size)    
        return target_size
    
    def make_mesh(self,writeto=None):
        """ construct a finite element mesh for the waveguide at current z (default 0).
        
        ARGS:
            writeto (str or None): filename for mesh (no extension). if None, no file is saved.  
        """
        m = self.make_mesh_bndry_ref(writeto)
        m.points = m.points[:,:2]
        return m

    def make_mesh_bndry_ref(self,writeto=None):
        """ construct a mesh with boundary refinement at material interfaces.
        
        ARGS:
            writeto (str or None): filename for mesh (no extension). if None, no file is saved.  
        """
        _scale = self.mesh_dist_scale
        _power = self.mesh_dist_power
        min_mesh_size = self.min_mesh_size
        max_mesh_size = self.max_mesh_size

        algo = 6
        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber('General.Terminal', 0)
            polygons = []
            for el in self.prim3Dgroups:
                if type(el) != list:
                    polygons.append(el.prim2D.make_poly(geom))
                else:
                    els = []
                    for _el in el:
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
                return self._compute_mesh_size(x,y,_scale=_scale,_power=_power,min_size=min_mesh_size,max_size=max_mesh_size)

            geom.env.removeAllDuplicates()
            geom.set_mesh_size_callback(callback)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
            if writeto is not None:
                gmsh.write(writeto+".msh")
                gmsh.clear()
            return mesh

    def assign_IOR(self):
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

    def plot_mesh(self, z=None, mesh=None, IOR_dict=None, alpha=0.1, ax=None, plot_points=True, verbose=True):
        transformed_mesh = mesh
        if mesh is None:
            z = 0 if z is None else z
            mesh0 = self.make_mesh_bndry_ref()
            transformed_mesh = self.transform_mesh(mesh0, 0, z)
        if IOR_dict is None:
            IOR_dict = self.assign_IOR()
        plot_mesh(transformed_mesh, IOR_dict, alpha, ax, plot_points, verbose=verbose)

    def plot_boundaries(self):
        import matplotlib.pyplot as plt
        for group in self.prim3Dgroups:
            if not type(group) == list:
                group = [group]
            for prim in group:
                p = prim.prim2D.points
                if hasattr(p[0][0], '__len__'):
                    for _p in p:
                        p2 = self.xp.zeros((_p.shape[0]+1,_p.shape[1]))
                        p2[:-1] = _p[:]
                        p2[-1] = _p[0]
                        plt.plot(p2.T[0], p2.T[1], color='0.5')
                else:
                    p2 = self.xp.zeros((p.shape[0]+1, p.shape[1]))
                    p2[:-1] = p[:]
                    p2[-1] = p[0]
                    plt.plot(p2.T[0], p2.T[1], color='k')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.axis('equal')
        plt.show()

    def _make_intersection_mesh(self,z,dz,writeto=None):
        """
        Not currently used. Construct a mesh around the union of Waveguide boundaries computed at z and z+dz.
        Returns both the mesh and a custom dictionary mapping regions of the mesh to refractive indices.
        To advance the refractive index profile from z to z+dz use self.advance_IOR() to update the dictionary.
        """
        _scale = self.mesh_dist_scale
        _power = self.mesh_dist_power
        min_mesh_size = self.min_mesh_size
        max_mesh_size = self.max_mesh_size

        IOR_dict={}

        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber('General.Terminal', 0)
            self.update(z)
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

            # Boolean subtraction of layers
            for i in range(len(elmnts)-1):
                group = elmnts[i]
                _group = elmnts[i+1]
                for j,fragroup in enumerate(group):
                    for _fragroup in _group:
                        idx = 0
                        _idx = 0
                        if isinstance(fragroup, list) and fragroup[0] is None:
                            idx = 1
                        if isinstance(_fragroup, list) and _fragroup[0] is None:
                            _idx = 1

                        if isinstance(fragroup, list):
                            if isinstance(_fragroup, list):
                                fragroup[idx:] = geom.boolean_difference(fragroup[idx:],_fragroup[_idx:],delete_first=True,delete_other=False)
                            else:
                                fragroup[idx:] = geom.boolean_difference(fragroup[idx:],_fragroup,delete_first=True,delete_other=False)
                        else:
                            if isinstance(_fragroup, list):
                                fragroup = geom.boolean_difference(fragroup,_fragroup[_idx:],delete_first=True,delete_other=False)
                            else:
                                fragroup = geom.boolean_difference(fragroup,_fragroup,delete_first=True,delete_other=False)

            # Add labels
            prim3Dgroups =[]
            for gr in self.prim3Dgroups:
                if isinstance(gr, list):
                    prim3Dgroups.append(gr)
                else:
                    prim3Dgroups.append([gr])

            for i,layer in enumerate(elmnts):
                for j,sublayer in enumerate(layer):
                    if isinstance(sublayer, list):
                        # lists contain two or three fragments, formed by the union of the 3D primitives evaled at z and z + dz
                        if len(sublayer)==2:
                            geom.add_physical(sublayer[0],prim3Dgroups[i][j].label+"2")
                            IOR_dict[prim3Dgroups[i][j].label+"2"] = prim3Dgroups[i][j].prim2D.n
                            geom.add_physical(sublayer[1],prim3Dgroups[i][j].label+"1")
                            if isinstance(self.prim3Dgroups[i-1], list):
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
                            if isinstance(self.prim3Dgroups[i-1], list):
                                IOR_dict[prim3Dgroups[i][j].label+"1"] = prim3Dgroups[i-1][j].prim2D.n
                            else:
                                IOR_dict[prim3Dgroups[i][j].label+"1"] = prim3Dgroups[i-1][0].prim2D.n

                        else:
                            raise Exception("wrong number of fragments produced by geom.boolean_fragments()")
                    else:
                        # if the sublayer is just a polygon, its structure is assumed to be fixed from z -> z + dz
                        if isinstance(self.prim3Dgroups[i], list):
                            geom.add_physical(layer,self.prim3Dgroups[i].label)
                            IOR_dict[self.prim3Dgroups[i].label] = self.prim3Dgroups[i].prim2D.n
                        else:
                            geom.add_physical(layer,self.prim3Dgroups[i].label)
                            IOR_dict[self.prim3Dgroups[i].label] = self.prim3Dgroups[i].prim2D.n
                        break
                
            # Mesh refinement callback
            def callback(dim,tag,x,y,z,lc):
                return self._compute_mesh_size(x,y,_scale=_scale,_power=_power,min_size=min_mesh_size,max_size=max_mesh_size)

            geom.env.removeAllDuplicates()
            geom.set_mesh_size_callback(callback)

            mesh = geom.generate_mesh(dim=2,order=2,algorithm=6)
            if writeto is not None:
                gmsh.write(writeto+".msh")
                gmsh.clear()
            return mesh,IOR_dict

    def assign_IOR(self):
        """Build a dictionary mapping material labels in the Waveguide mesh 
        to the corresponding refractive index value."""
        for p in self.prim3Dgroups:
            if isinstance(p, list):
                for _p in p:
                    if _p.label in self.IOR_dict:
                        continue
                    self.IOR_dict[_p.label] = _p.prim2D.n
            else:
                if p.label in self.IOR_dict:
                    continue
                self.IOR_dict[p.label] = p.prim2D.n  
        return self.IOR_dict

    def plot_mesh(self,z=None,mesh=None,IOR_dict=None,alpha=0.1,ax=None,plot_points=True,verbose=True):
        """ plot a mesh and associated refractive index distribution
        
        ARGS:
            z: the z coordinate for the mesh we want to plot. if None, z=0.
            mesh: the mesh to be plotted. if None, we auto-compute a mesh using default values
            IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
            alpha: transparency of mesh lines
            ax: optionally, pass in a matplotlib axis for the plotter
            plot_points: controls if the mesh nodes will be plotted
        """
        transformed_mesh = mesh
        if mesh is None:
            z = 0 if z is None else z
            mesh0 = self.make_mesh_bndry_ref()
            transformed_mesh = self.transform_mesh(mesh0,0,z)

        if IOR_dict is None:
            IOR_dict = self.assign_IOR()

        plot_mesh(transformed_mesh,IOR_dict,alpha,ax,plot_points,verbose=verbose)

    def plot_boundaries(self):
        """Plot boundaries of all prim3Dgroups. For unions, plot all with lighter color."""
        for group in self.prim3Dgroups:
            if not isinstance(group, list):
                group = [group]
            for prim in group:
                p = prim.prim2D.points
                if hasattr(p[0][0],'__len__'):
                    for _p in p:
                        p2 = xp.zeros((_p.shape[0]+1, _p.shape[1]))
                        p2[:-1] = _p[:]
                        p2[-1] = _p[0]
                        plt.plot(p2.T[0],p2.T[1],color='0.5')
                else:
                    p2 = xp.zeros((p.shape[0]+1, p.shape[1]))
                    p2[:-1] = p[:]
                    p2[-1] = p[0]
                    plt.plot(p2.T[0],p2.T[1],color='k')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.axis('equal')
        plt.show()
    
    def transform_unvec(self,x0,y0,z0,z):
        """Unvectorized version of transform(); present for reference."""
        if self.z_invariant:
            return x0,y0

        pt = xp.array([x0, y0])
        self.update(z0)
        preserved_prim3Ds = [p for p in self.prim3Dsflat if p.preserve_shape]
        eps = 1e-12
        
        for p in preserved_prim3Ds:
            if p.prim2D.boundary_dist(x0,y0)<=eps:
                return p.transform_point_inside(x0,y0,z0,z)

        boundary_points = xp.zeros((len(preserved_prim3Ds), 2))
        next_boundary_points = xp.zeros_like(boundary_points)

        for i,p in enumerate(preserved_prim3Ds):
            bp = p.prim2D.nearest_boundary_point(x0,y0)
            boundary_points[i,:] = bp
            next_boundary_points[i,:] = p.transform_point_inside(bp[0],bp[1],z0,z)

        prim_pairs = list(combinations(xp.arange(len(preserved_prim3Ds)), 2))
        avg_dists = [ 0.5*(dist(boundary_points[i],pt)+dist(boundary_points[j],pt)) for (i,j) in prim_pairs]
        idx_min_dist = xp.argmin(avg_dists)
        i,j = prim_pairs[idx_min_dist]

        bp_sep = dist(boundary_points[i],boundary_points[j])
        next_bp_sep = dist(next_boundary_points[i],next_boundary_points[j])
        _scale = next_bp_sep/bp_sep
        
        edge_vec1 = pt - boundary_points[i]
        edge_vec2 = pt - boundary_points[j]

        boundary_edge = boundary_points[i] - boundary_points[j]
        next_boundary_edge = next_boundary_points[i] - next_boundary_points[j]

        angl = xp.arctan2(boundary_edge[1], boundary_edge[0])
        next_angl = xp.arctan2(next_boundary_edge[1], next_boundary_edge[0])
        rot = next_angl-angl

        np1 = rotate(edge_vec1*_scale,rot) + next_boundary_points[i]
        np2 = rotate(edge_vec2*_scale,rot) + next_boundary_points[j]

        new_point = np1 if dist(pt, boundary_points[i]) <= dist(pt, boundary_points[j]) else np2
        return new_point[0],new_point[1]
    
    def transform(self,x0,y0,z0,z):
        """ spatial transformation that should work with most meshes. kinda slow to compute currently, maybe could be sped up.
        general idea: for a point (x0,y0) find the two nearest primitives at z0. this point, plus the two closest points to 
        to it, each constrained to lie on a primitive boundary, define a triangle. At z, the primitive boundaries move, and so 
        two of the triangle vertices move. this transformation places the third point of the triangle, which is (x,y), to
        preserve triangle similarity.

        ARGS:
            x0 : (scalar or vector, if self.vectorized_transform) x coord. at z0
            y0 : (scalar or vector, if self.vectorized_transform) y coord. at z0
            z0 : reference z coord.
            z : new z coord.

        RETURNS:
            (tuple) : a tuple containing:

                - x (scalar or vector): the transformed x coordinate at z.
                - y (scalar or vector): the transformed y coordinate at z.
        """
        if not isinstance(x0, xp.ndarray):
            x0 = xp.array([x0])
            y0 = xp.array([y0])

        if self.z_invariant:
            return x0,y0

        pt = xp.array([x0, y0]).T
        if z0 != 0:
            self.update(z0)

        preserved_prim3Ds = [p for p in self.prim3Dsflat if p.preserve_shape]
        boundary_points = xp.zeros((len(x0), len(preserved_prim3Ds), 2))
        next_boundary_points = xp.zeros_like(boundary_points)

        for i,p in enumerate(preserved_prim3Ds):
            bps = p.prim2D.nearest_boundary_point(x0,y0)
            boundary_points[:,i,0] = bps[0]
            boundary_points[:,i,1] = bps[1]
            nbps = xp.array(p.transform_point_inside(bps[0], bps[1], z0, z)).T
            next_boundary_points[:,i,0] = nbps[:,0]
            next_boundary_points[:,i,1] = nbps[:,1]

        prim_pairs = xp.array(list(combinations(xp.arange(len(preserved_prim3Ds)), 2)))
        avg_dists = xp.array([0.5 * (dist(boundary_points[:, i], pt, axis=1) + dist(boundary_points[:, j], pt, axis=1)) for (i, j) in prim_pairs])
        idx_min_dist = xp.argmin(avg_dists, axis=0)
        ijs = xp.array(prim_pairs[idx_min_dist])

        idxrange = xp.arange(x0.shape[0])
        bps0 = boundary_points[idxrange,ijs[:,0]]
        bps1 = boundary_points[idxrange,ijs[:,1]]

        nbps0 = next_boundary_points[idxrange,ijs[:,0]]
        nbps1 = next_boundary_points[idxrange,ijs[:,1]]

        bp_sep = dist(bps0,bps1,axis=1)
        next_bp_sep = dist(nbps0,nbps1,axis=1)
        _scale = next_bp_sep/bp_sep

        edge_vec1 = pt - bps0
        edge_vec2 = pt - bps1
        boundary_edge = bps0-bps1
        next_boundary_edge = nbps0-nbps1

        angl = xp.arctan2(boundary_edge[:, 1], boundary_edge[:, 0])
        next_angl = xp.arctan2(next_boundary_edge[:, 1], next_boundary_edge[:, 0])
        rot = next_angl-angl
        np1 = rotate(edge_vec1*_scale[:,None],rot) + nbps0
        np2 = rotate(edge_vec2*_scale[:,None],rot) + nbps1
        npsx,npsy = (0.5*(np1+np2)).T
        eps=1e-12

        # check inside masked regions
        for p in preserved_prim3Ds:
            _in = p.prim2D.boundary_dist(x0,y0) <= eps
            xi,yi = p.transform_point_inside(x0[_in],y0[_in],z0,z)
            npsx[_in],npsy[_in] = xi,yi
        
        if len(npsx)==1:
            return npsx[0],npsy[0]
        return npsx,npsy

    def deriv_transform(self,x0,y0,z0,z):
        """ compute the derivative of the transformation law. default 
        implementation uses finite differences, but an explit law 
        can be written for inheriting classes (e.g. PhotonicLantern).

        ARGS:
            x0 : (scalar or vector, if self.vectorized_transform) x coord. at z0
            y0 : (scalar or vector, if self.vectorized_transform) y coord. at z0
            z0 : reference z coord.
            z : new z coord.
        RETURNS:
            x : the transformed x coordinate at z
            y : the transformed y coordinate at z
        """
        if self.z_invariant:
            return xp.zeros_like(x0), xp.zeros_like(y0)
        if self.vectorized_transform:
            x1,y1 = self.transform(x0,y0,z0,z-self.fd_eps)
            x2,y2 = self.transform(x0,y0,z0,z+self.fd_eps)
            return 0.5*(x2-x1)/self.fd_eps , 0.5*(y2-y1)/self.fd_eps
        else:
            dx, dy = xp.zeros_like(x0), xp.zeros_like(y0)
            for i in range(dx.shape[0]):
                x1,y1 = self.transform(x0[i],y0[i],z0,z-self.fd_eps)
                x2,y2 = self.transform(x0[i],y0[i],z0,z+self.fd_eps)
                dx[i] = 0.5*(x2-x1)/self.fd_eps
                dy[i] = 0.5*(y2-y1)/self.fd_eps
            return dx,dy

    def transform_mesh(self,mesh0,z0,z,mesh=None):
        """ use the transformation law to create a new, transformed 
        mesh from the reference mesh.
        
        ARGS:
            mesh0: the reference mesh (meshio object).
            z0: the z coordinate corresponding to mesh0 (typically 0).
            z: the desired z coordinate of the transformed mesh.
            mesh: a mesh object to store the new mesh. if None, a new mesh
            object is generated.
        """
        if self.z_invariant or z0==z:
            return mesh0
        if mesh is None:
            mesh = copy.deepcopy(mesh0)

        xp0 = mesh0.points[:,0]
        yp0 = mesh0.points[:,1]

        if self.vectorized_transform:
            xp_, yp_ = self.transform(xp0, yp0, z0, z)
        else:
            xp_ = xp.zeros_like(xp0)
            yp_ = xp.zeros_like(yp0)
            for i in range(xp0.shape[0]):
                xp_[i], yp_[i] = self.transform(xp0[i], yp0[i], z0, z)
        mesh.points[:, 0] = xp_
        mesh.points[:, 1] = yp_
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
    
    def isolate(self,k):
        """ create a refractive index dictionary that sets all channels except one to the background index.
        'channels' are assumed to be stored in the last list of self.prim3Dgroups. 
        override this if your waveguide is different.
        
        ARGS:
            k: index of the channel to isolate
        """
        self.assign_IOR()
        IOR_dict = copy.copy(self.IOR_dict)
        if isinstance(self.prim3Dgroups[-2], list):
            nb = self.prim3Dgroups[-2][0].n
        else:
            nb = self.prim3Dgroups[-2].n
        for i in range(len(self.prim3Dgroups[-1])):
            p = self.prim3Dgroups[-1][i]
            if i != k:
                IOR_dict[p.label] = nb
        return IOR_dict

class CircularStepIndexFiber(Waveguide):
    ''' generic class for a straight, circular step index fiber '''
    vectorized_transform = True
    recon_midpts = False
    z_invariant = True

    def __init__(self,rcore,rclad,ncore,nclad,core_res=16,clad_res=32):
        """ initialize a circular step index fiber waveguide

        ARGS: 
            rcore: radius of core
            rclad: radius of cladding
            ncore: refractive index of the core
            nclad: refractive index of the cladding
            core_res: how many line segments to divide the core boundary into, default 16
            clad_res: how many line segments to divide the cladding boundary into, default 32
        """
        core = Pipe(ncore,"core",core_res,rcore,(0,0))
        clad = Pipe(nclad,"clad",clad_res,rclad,(0,0))
        core.mesh_size = 2*xp.pi*rcore/core_res
        clad.skip_refinement = True
        els = [clad,core]        
        super().__init__(els)

class RectangularStepIndexFiber(Waveguide):
    ''' generic class for a straight, rectangular step index fiber '''
    vectorized_transform = True
    recon_midpts = False
    z_invariant = True

    def __init__(self,xw,yw,xw_clad,yw_clad,ncore,nclad,core_mesh_size=None,clad_mesh_size=None):
        """ intialize a waveguide with a rectangular core and cladding.

        ARGS:
            xw: width of the rectangular core
            yw: height of the rectangular core
            xw_clad: width of the cladding
            yw_clad: height of the cladding
            ncore: refractive index of the core
            nclad: refractive index of the cladding
            core_mesh_size: target mesh size inside the core; default is min(xw,yw)/10
            clad_mesh_size: target mesh size in the cladding; default is min(xw_clad,yw_clad)/10
        """
        core = BoxPipe(ncore,"core",xw,yw)
        clad = BoxPipe(nclad,"clad",xw_clad,yw_clad)
        core.mesh_size = min(xw,yw)/10 if core_mesh_size is None else core_mesh_size
        clad.mesh_size = min(xw_clad,yw_clad)/10 if clad_mesh_size is None else clad_mesh_size
        clad.skip_refinement = True
        els = [clad,core]
        super().__init__(els)


#---------------------- PhotonicLantern and friends ------------------------------

class PhotonicLantern(Waveguide):
    ''' generic class for photonic lanterns '''
    recon_midpts = False
    linear=True

    def __init__(self,core_pos,rcores,rclad,rjack,ncores,nclad,njack,z_ex,taper_factor,core_res=30,clad_res=60,jack_res=30,core_mesh_size=None,clad_mesh_size=None):
        ''' initialize a photonic lantern waveguide.

        ARGS: 
            core_pos: an Nx2 array of (x,y) core positions at z=0
            rcores: an array of core radii at z=0
            rclad: the cladding radius at z=0
            rjack: the jacket radius at z=0 (this sets the outer simulation boundary)
            ncores: an array of refractive indices for the cores
            nclad: the cladding refractive index
            njack: the jacket refractive index
            z_ex: the lantern length
            taper_factor: the amount the lantern scales by, going from z=0 -> z=z_ex
            core_res: number of line segments to resolve each core-cladding boundary with
            clad_res: number of line segments to resolve the cladding-jacket boundary
            jack_res: number of line segments to resolve the outer jacket boundary
            core_mesh_size: target side length for triangles inside a core. defaults to the size set by core_res
            clad_mesh_size: target side length for triangles inside the cladding. defaults to the size set by clad_res
        '''
        if core_mesh_size is None:            
            core_mesh_size = 2 * xp.pi * xp.mean(xp.asarray(rcores)) / core_res
        if clad_mesh_size is None:
            clad_mesh_size = 2*xp.pi*rclad/clad_res

        taper_func = lambda z: (taper_factor - 1)/z_ex * z + 1
        self.taper_func = taper_func
        self.taper_factor = taper_factor

        cores = []
        def rfunc(r):
            return lambda z: taper_func(z) * r
        def cfunc(c):
            return lambda z: (taper_func(z)*c[0], taper_func(z)*c[1])

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
        self.min_mesh_size = min(self.min_mesh_size,core_mesh_size,clad_mesh_size)

    def transform(self,x0,y0,z0,z):
        scale =  self.taper_func(z)/self.taper_func(z0)
        return x0  * scale , y0  * scale

    def deriv_transform(self,x0,y0,z0,z):
        return x0*(self.taper_factor-1)/self.z_ex , y0*(self.taper_factor-1)/self.z_ex

    def isolate_isect(self,k,d):
        IOR_dict = copy.copy(d)
        for i in range(len(self.prim3Dgroups[-1])):
            if i != k:
                IOR_dict["core"+str(i)+"0"] = IOR_dict["cladding2"]
                IOR_dict["core"+str(i)+"1"] = IOR_dict["cladding2"]
                IOR_dict["core"+str(i)+"2"] = IOR_dict["cladding2"]
        return IOR_dict

class TestPhotonicLantern(PhotonicLantern):
    """ shorthand for doctesting """
    def __init__(self):
        taper_factor = 8.
        rcore = 2.2 / taper_factor
        rclad = 10
        rjack = 30
        z_ex = 40000
        nclad = 1.444
        ncore = nclad + 8.8e-3
        njack = nclad - 5.5e-3
        core_offset = rclad * 2/3
        core_pos = xp.array([[0,0]] + [[core_offset*xp.cos(i*2*xp.pi/5), core_offset*xp.sin(i*2*xp.pi/5)] for i in range(5)])
        core_res = 30
        clad_res = 60
        jack_res = 30
        clad_mesh_size = 1.0
        core_mesh_size = 0.1
        rcores = [rcore] * 6
        ncores = [ncore] * 6
        super().__init__(core_pos,rcores,rclad,rjack,ncores,nclad,njack,z_ex,taper_factor,core_res,clad_res,jack_res,core_mesh_size,clad_mesh_size)

# ---------------------- Dicoupler, Tricoupler, PlanarTricoupler, OAMLantern ----------------------

class Dicoupler(Waveguide):
    ''' generic class for 2x2 directional couplers made of pipes '''
    recon_midpts = True

    def __init__(self,rcore1,rcore2,ncore1,ncore2,dmax,dmin,nclad,coupling_length,a,core_res,core_mesh_size,clad_mesh_size):
        """ initialize a directional coupler waveguide.

        ARGS:
            rcore1: the core radius of the left single mode channel
            rcore2: the core radius of the right single mode channel
            ncore1: the index of the left channel
            ncore2: the index of the right channel
            dmax: the starting core separation
            dmin: the minimum core separation
            nclad: the cladding index
            coupling_length: the approximate length in z over which the core separation is dmin
            a: the approximate bend length
            core_res: the # of segments used to resolve the core-cladding interface
            core_mesh_size: the target mesh size inside cores
            clad_mesh_size: the target mesh size inside the cladding
        """
        z_ex = coupling_length * 2

        def c2func(z):
            if z <= z_ex/2:
                b = blend(z,z_ex/4-a/2,a)
                return xp.array([dmin/2,0])*b + xp.array([dmax/2,0])*(1-b)
            else:
                b = blend(z,3*z_ex/4+a/2,a)
                return xp.array([dmax/2,0])*b + xp.array([dmin/2,0])*(1-b)

        def c1func(z): return -c2func(z)

        def dfunc(z): return c2func(z)[0]-c1func(z)[0]  # inter core spacing function
        
        self.c1func = c1func
        self.c2func = c2func
        self.dfunc = dfunc
        self.eps = 1e-12

        cladding = BoxPipe(nclad,"cladding",lambda z: self.dfunc(z)*6,lambda z: self.dfunc(z)*4)
        cladding.mesh_size = clad_mesh_size
        cladding.skip_refinement = True
        cladding.preserve_shape = False

        core1 = Pipe(ncore1,"core1",core_res,lambda z: rcore1,c1func)
        core1.mesh_size = core_mesh_size
        core1.preserve_shape = True

        core2 = Pipe(ncore2,"core2",core_res,lambda z: rcore2,c2func)
        core2.mesh_size = core_mesh_size
        core2.preserve_shape = True

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

        x1 = xp.where(
            xp.logical_and(c1_0[0] + self.rcore1 + self.eps < x0, x0 < c2_0[0] - self.rcore2 - self.eps),
            x0 * xscale,
            x0
        )
        x2 = xp.where(x1 <= c1_0[0] + self.rcore1 + self.eps, x1 - dd, x1)
        x3 = xp.where(x2 >= c2_0[0] - self.rcore2 - self.eps, x2 + dd, x2)
        return x3,y0

    def plot_paths(self):
        """Plot the single-mode channel paths."""
        zs = xp.linspace(0, self.z_ex, 400)
        c1s = [self.c1func(float(z))[0] for z in zs]
        c2s = [self.c2func(float(z))[0] for z in zs]
        plt.plot(zs,c1s,label="channel 1")
        plt.plot(zs,c2s,label="channel 2")
        plt.xlabel(r"$z$")
        plt.ylabel(r"$x$")
        plt.legend(loc='best',frameon=False)
        plt.show()
class Tricoupler(Waveguide):
    ''' generic class for symmetric equilateral tricouplers made of pipes '''
    recon_midpts = True

    def __init__(self,rcore,ncore,dmax,dmin,nclad,coupling_length,a,core_res,core_mesh_size,clad_mesh_size):
        """ initialize a tricoupler waveguide.

        ARGS:
            rcore: the core radius, shared for all single mode channels
            ncore: the refractive index for all single mode channels
            dmax: the diameter of the circle containing the center of each channel, at max separation
            dmin: the diameter of the circle containing the center of each channel, at min separation
            nclad: the cladding index
            coupling_length: the approximate length in z over which the core separation is dmin
            a: the approximate bend length
            core_res: the # of segments used to resolve the core-cladding interface
            core_mesh_size: the target mesh size inside cores
            clad_mesh_size: the target mesh size inside the cladding
        """
        z_ex = coupling_length * 2

        def c1func(z):
            if z <= z_ex/2:
                b = blend(z,z_ex/4-a/2,a)
                return xp.array([dmin/2,0])*b + xp.array([dmax/2,0])*(1-b)
            else:
                b = blend(z,3*z_ex/4+a/2,a)
                return xp.array([dmax/2,0])*b + xp.array([dmin/2,0])*(1-b)

        def c2func(z):
            c = c1func(z)
            return c[0]*xp.cos(2*xp.pi/3), c[0]*xp.sin(2*xp.pi/3)            
        
        def c3func(z):
            c = c1func(z)
            return c[0]*xp.cos(4*xp.pi/3), c[0]*xp.sin(4*xp.pi/3)

        self.c1func = c1func
        self.c2func = c2func
        self.c3func = c3func

        cladding = BoxPipe(nclad, "cladding", lambda z: c1func(z)[0]*8, lambda z: c1func(z)[0]*8)
        cladding.mesh_size = clad_mesh_size
        cladding.skip_refinement = True
        cladding.preserve_shape = False

        core1 = Pipe(ncore,"core1",core_res,lambda z: rcore,c1func)
        core1.mesh_size = core_mesh_size
        core1.preserve_shape = True

        core2 = Pipe(ncore,"core2",core_res,lambda z: rcore,c2func)
        core2.mesh_size = core_mesh_size
        core2.preserve_shape = True

        core3 = Pipe(ncore,"core3",core_res,lambda z: rcore,c3func)
        core3.mesh_size = core_mesh_size
        core3.preserve_shape = True


        els = [cladding,[core1,core2,core3]]
        self.rcore = rcore
        super().__init__(els)
        self.z_ex = z_ex


    def plot_paths(self):
        """Plot the single-mode channel paths"""
        zs = xp.linspace(0, self.z_ex, 400)
        # Always use xp.array for backend-agnostic, but convert to numpy before plotting if using JAX
        c1s = xp.array([self.c1func(float(z)) for z in zs])
        c2s = xp.array([self.c2func(float(z)) for z in zs])
        c3s = xp.array([self.c3func(float(z)) for z in zs])

        # For matplotlib, always convert to numpy arrays (JAX DeviceArrays won't plot!)
        zs_np = xp.asarray(zs)
        c1s_np = xp.asarray(c1s)
        c2s_np = xp.asarray(c2s)
        c3s_np = xp.asarray(c3s)

        fig,axs = plt.subplots(2,1,sharex=True)
        axs[0].plot(zs_np, c1s_np[:, 0], label="channel 1")
        axs[0].plot(zs_np, c2s_np[:, 0], label="channel 2")
        axs[0].plot(zs_np, c3s_np[:, 0], label="channel 3")

        axs[1].plot(zs_np, c1s_np[:, 1], label="channel 1")
        axs[1].plot(zs_np, c2s_np[:, 1], label="channel 2")
        axs[1].plot(zs_np, c3s_np[:, 1], label="channel 3")

        axs[0].set_ylabel(r"$x$")
        axs[1].set_ylabel(r"$y$")
        axs[1].set_xlabel(r"$z$")
        axs[0].legend(loc="best", frameon=False)
        plt.subplots_adjust(hspace=0,wspace=0)
        plt.show()
class PlanarTricoupler(Tricoupler):
    ''' Tricoupler with 2D channel paths '''
    recon_midpts = True
    linear=False

    def __init__(self, rcore_center, rcore_outer, ncore, dmax, dmin, nclad,
                 coupling_length, a, core_res, core_mesh_size, clad_mesh_size):
        """ initialize a planar tricoupler waveguide.

        ARGS:
            rcore_center: the core radius of the center channel
            rcore_outer: the core radius of the outer channel
            ncore: the refractive index for all single mode channels
            dmax: the distance between the left and right channels, at max separation
            dmin: the distance between the left and right channels, at min separation
            nclad: the cladding index
            coupling_length: the approximate length in z over which the core separation is dmin
            a: the approximate bend length
            core_res: the # of segments used to resolve the core-cladding interface
            core_mesh_size: the target mesh size inside cores
            clad_mesh_size: the target mesh size inside the cladding
        """
        z_ex = coupling_length * 2
        def c1func(z):
            if z <= z_ex/2:
                b = blend(z,z_ex/4-a/2,a)
                return xp.array([dmin/2,0])*b + xp.array([dmax/2,0])*(1-b)
            else:
                b = blend(z,3*z_ex/4+a/2,a)
                return xp.array([dmax/2,0])*b + xp.array([dmin/2,0])*(1-b)
        def c2func(z):
            c = c1func(z)
            return -c[0],c[1]
        
        def c3func(z):
            return 0,0

        self.c1func = c1func
        self.c2func = c2func
        self.c3func = c3func

        cladding = BoxPipe(nclad, "cladding", lambda z: c1func(z)[0]*8, lambda z: c1func(z)[0]*8)
        cladding.mesh_size = clad_mesh_size
        cladding.skip_refinement = True
        cladding.preserve_shape = False

        core1 = Pipe(ncore,"core1",core_res,lambda z: rcore_outer,c1func)
        core1.mesh_size = core_mesh_size
        core1.preserve_shape = True

        core2 = Pipe(ncore,"core2",core_res,lambda z: rcore_outer,c2func)
        core2.mesh_size = core_mesh_size
        core2.preserve_shape = True

        core3 = Pipe(ncore,"core3",core_res,lambda z: rcore_center,c3func)
        core3.mesh_size = core_mesh_size
        core3.preserve_shape = True

        els = [cladding,[core1,core2,core3]]
        self.rcore_center = rcore_center
        self.rcore_outer = rcore_outer

        Waveguide.__init__(self,els)
        self.z_ex = z_ex
    
class OAMPhotonicLantern(PhotonicLantern):
    ''' OAM photonic lantern using ring core '''
    vectorized_transform = True
    recon_midpts = False

    def __init__(self, ring_radius, ring_width, rcores, rjack, ncores,
                 nclad, njack, z_ex, taper_factor, core_res, clad_res_outer,
                 clad_res_inner, jack_res, core_mesh_size=0.05, clad_mesh_size=0.2, inner_clad_mesh_size=0.2):

        ''' ARGS: 
            ring_radius: radius of cladding annulus (avg)
            ring_width: width of cladding annulus
            rcores: an array of core radii at z=0. the first value is placed in the middle; the rest spaced evenly in the ring
            rjack: the jacket radius at z=0 (this sets the outer simulation boundary)
            ncores: an array of refractive indices for the cores
            nclad: the cladding annulus refractive index
            njack: the jacket refractive index
            z_ex: the lantern length
            taper_factor: the amount the lantern scales by, going from z=0 -> z=z_ex
            core_res: the number of line segments to resolve each core-cladding boundary with
            clad_res_outer: the number of line segments to resolve the outer cladding annulus boundary
            clad_res_inner: the number of line segments to resolve the inner cladding annulus boundary
            jack_res: the number of line segments to resolve the outer jacket boundary
            core_mesh_size: the target side length for triangles inside a lantern core, away from the boundary
            clad_mesh_size: the target side length for triangles inside the lantern cladding, away from the boundary
        '''
        N_outer = len(rcores)
        core_pos = xp.array([[ring_radius*xp.cos(i*2*xp.pi/N_outer),
                              ring_radius*xp.sin(i*2*xp.pi/N_outer)] for i in range(N_outer)])

        taper_func = lambda z: (taper_factor - 1)/z_ex * z + 1
        self.taper_func = taper_func

        cores = []
        def rfunc(r): return lambda z: taper_func(z) * r
        def cfunc(c): return lambda z: (taper_func(z)*c[0], taper_func(z)*c[1])

        for k,(c,r,n) in enumerate(zip(core_pos,rcores,ncores)):
            cores.append(Pipe(n,"core"+str(k),core_res,rfunc(r),cfunc(c)))
            cores[k].mesh_size = core_mesh_size

        cladrfunc_outer = lambda z: taper_func(z)*(ring_radius+ring_width/2)
        cladrfunc_inner = lambda z: taper_func(z)*(ring_radius-ring_width/2)
        cladcfunc = lambda z: (0,0)

        _clad_outer = Pipe(nclad,"cladding",clad_res_outer,cladrfunc_outer,cladcfunc)
        _clad_outer.mesh_size = clad_mesh_size
        _clad_inner = Pipe(njack,"inner_cladding",clad_res_inner,cladrfunc_inner,cladcfunc)
        _clad_inner.mesh_size = inner_clad_mesh_size

        jackrfunc = lambda z: taper_func(z)*rjack
        jackcfunc = lambda z: (0,0)
        _jack = Pipe(njack,"jacket",jack_res,jackrfunc,jackcfunc)
        _jack.skip_refinement = True

        els = [_jack,_clad_outer,[_clad_inner]+cores]
        
        Waveguide.__init__(self,els)
        self.z_ex = z_ex



