from juliacall import Main as jl
import numpy as np
import os,cbeam
from juliacall import Pkg as jlPkg

jlPkg.activate(os.path.dirname(cbeam.__file__)+"/FEval")
jl.seval("using FEval")

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
    return jl.FEval.construct_tritree(points,connections+1)

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
    jl_idx = jl.FEval.query(point,tree)
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
    if point.ndim == 2:
        return np.array(jl.FEval.evaluate(point[:,:2],field,tree))
    return np.array(jl.FEval.evaluate(point,field,tree))

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
    return jl.FEval.evaluate(pointsx,pointsy,field,tree)

def update_tree(tree,rescale_factor):
    jl.FEval.update_tritree(tree,rescale_factor)
    return

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
    return np.array(jl.FEval.transverse_gradient(field,tris,points))

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