from juliacall import Main as jl
from .backend import get_xp
import os,cbeam
import numpy as _np

from juliacall import Main as jl

# Load Pkg natively using Julia syntax
jl.seval("using Pkg")
Pkg = jl.Pkg
xp = get_xp()


def _host(a, dtype=None):
    """Coerce an array (possibly a JAX device array) to a contiguous host numpy
    array before handing it to Julia.  PythonCall wraps a numpy ndarray as a
    zero-copy ``PyArray``; a JAX array instead arrives as ``PyIterable{Any}``
    and fails method dispatch in FEval.jl.

    With ``dtype=None`` the value is promoted to the dtype FEval.jl expects:
    float64 for real input, complex128 for complex input (its ``evaluate``
    methods are ``T<:Union{Float64,ComplexF64}``); integer/bool arrays such as
    triangle-connectivity are left as they are.
    """
    a = _np.asarray(a)
    if dtype is None:
        if _np.iscomplexobj(a):
            dtype = _np.complex128
        elif _np.issubdtype(a.dtype, _np.floating):
            dtype = _np.float64
    return _np.ascontiguousarray(a, dtype=dtype)

# ===== ADD THIS: Load the FEval Julia module =====
_cbeam_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_feval_jl = os.path.join(_cbeam_root, "cbeam", "FEval", "src", "FEval.jl")
# raw"..." so the path is not run through Julia's string-escape rules: on
# Windows os.path.join yields backslashes and e.g. "...\Users\..." fails to
# parse as an invalid \U unicode escape.
jl.seval(f'include(raw"{_feval_jl}")')
# ================================================


def create_tree(points,connections):
    """ from an array of mesh points and an index array of (quadratic) triangle connections, 
    construct a bounding volume hierarchy (BVH) tree, which will be used to evaluate fields
    define on the mesh nodes.
    
    ARGS:
        points: an array of the (x,y) positions of the mesh nodes, dimension N x 2 for N nodes.
        connections: an array containing each triangle in the mesh; each triangle is represented
                     as 6 indices, corresponding to 6 points
    RETURNS:
        bvhtree: the BVH tree for the given mesh points and connections.
    """
    return jl.FEval.construct_tritree(_host(points, dtype=_np.float64),
                                      _host(connections) + 1)

def create_tree_from_mesh(mesh):
    """ create a BVH tree directly from a finite element mesh object.
    
    ARGS:
        mesh: a meshio object representing a finite element mesh
    RETURNS:
        bvhtree: the BVH tree for the given mesh points and connections.
    """
    return jl.FEval.construct_tritree(mesh.points,mesh.cells[1].data+1)

def sort_mesh(mesh):
    """ create a BVH tree for ``mesh`` and pass it into to ``mesh.tree`` """
    mesh.tree = create_tree_from_mesh(mesh)
    return mesh

def query(point,tree):
    """ find the index of the triangle in the mesh that contains the given point. 
    
    ARGS:
        point: an array [x,y] corresponding to the query point..
        tree: the BVH tree for the mesh of interest.
    RETURNS:
        (int): the index of the triangle containing point in the mesh.
    """
    jl_idx = jl.FEval.query(_host(point, dtype=_np.float64), tree)
    return jl_idx-1

def evaluate(point,field,tree):
    """ evaluate a field sampled over a finite element mesh at a given point.
    
    ARGS:
        point: an [x,y] point, or an Nx2 array of points
        field: a real-valued field represented on a finite-element mesh
        tree: a BVH tree that stores the triangles of field's finite-element mesh.
    RETURNS:
        (float or vector): the field evaluated at point(s)
    """
    point = _host(point, dtype=_np.float64)
    field = _host(field)                       # float64 or complex128
    if point.ndim == 2:
        return xp.array(jl.FEval.evaluate(point[:,:2], field, tree))
    return xp.array(jl.FEval.evaluate(point, field, tree))

def resample(field,mesh,newmesh):
    """ resample a finite element field onto a new mesh
    
    ARGS: 
        field: the finite element field to be resampled.
        mesh: the finite element mesh on which <field> is defined.
        newmesh: the new finite element mesh <field> should be sampled on.
    """
    tree = create_tree_from_mesh(mesh)
    return evaluate(newmesh.points,field,tree)

def evaluate_grid(pointsx,pointsy,field,tree):
    """ evaluate a field defined over a finite element mesh on a cartesian grid.
    
    ARGS:
        pointsx: a 1D array of x points
        pointsy: a 1D array of y points
        field: a real-valued field represented on a finite-element mesh
        tree: a BVH tree that stores the triangles of field's finite-element mesh

    RETURNS:
        (array): a 2D array corresponding to field, evaluated on the grid.
    """
    pointsx = _host(pointsx, dtype=_np.float64)
    pointsy = _host(pointsy, dtype=_np.float64)
    field   = _host(field)                     # float64 or complex128
    return jl.FEval.evaluate(pointsx,pointsy,field,tree)

def update_tree(tree,rescale_factor):
    jl.FEval.update_tritree(tree,rescale_factor)

def evaluate_func(field,tree):
    """ return a (julia) function of the point [x,y] corresponding to a given FE field """
    return jl.FEval.evaluate_func(field,tree)

def transverse_gradient(field,tris,points):
    """ compute the gradient of a real-valued finite element field with respect to x,y
    
    ARGS:
        field: a real-valued field represented on a finite-element mesh
        tris: an Nx6 array of triangles, each row contains the 6 indices of one (quadratic) triangle element
        points: an Mx2 array of [x,y] points representing the mesh nodes.
    RETURNS:
        (array): dimensions are dim(field) x 2; the last axis contains the x and y derivatives. 
    """
    field  = _host(field)                      # float64 or complex128
    tris   = _host(tris)
    points = _host(points, dtype=_np.float64)
    return xp.array(jl.FEval.transverse_gradient(field, tris, points))

def get_triangles(mesh):
    """ get the triangles in a mesh. 
    
    ARGS:
        mesh: a meshio object, e.g. the output of Waveguide.make_mesh()
    RETURNS:
        (array): Nx6 array of triangles, each row contains the 6 indices of one (quadratic) triangle element.
        The indices each identify a point in the mesh.   
    """
    return mesh.cells[1].data

def get_points(mesh):
    """ get the point array in a mesh. 

    ARGS:
        mesh: a meshio object, e.g. the output of Waveguide.make_mesh()
    RETURNS:
        (array): Mx2 or Mx3 array of point coordinates; first two columns are 
        x and y coordinates; you can ignore the third columnn if it exists.
    """
    return mesh.points