from juliacall import Main as jl
from juliacall import Pkg as jlPkg
import numpy as np

jlPkg.activate("FEval") 
jl.seval("using FEval")

def create_tree(points,connections):
    """ from an array of mesh points and an index array of (quadratic) triangle connections, construct a BVH tree """
    return jl.FEval.construct_tritree(points,connections+1)

def query(point,tree):
    """ find the index of the triangle that contains the given point. """
    jl_idx = jl.FEval.query(point,tree)
    return jl_idx-1

def get_idxs_and_weights(new_points,tree):
    triidxs,interpweights = jl.FEval.get_interp_weights(new_points,tree)
    return np.array(triidxs)-1,np.array(interpweights)

def evaluate(point,field,tree):
    """ evaluate a field sampled over a finite element mesh at a given point.
    ARGS:
        point: an [x,y] point, or an Nx2 array of points
        field: a real-valued field represented on a finite-element mesh
        tree: a BVH tree that stores the triangles of field's finite-element mesh
    """
    return jl.FEval.evaluate(point,field,tree)

def evaluate_grid(pointsx,pointsy,field,tree):
    """ evaluate a field sampled over a finite element mesh on a cartesian grid.
    ARGS:
        pointsx: a 1D array of x points
        pointsy: a 1D array of y points
        field: a real-valued field represented on a finite-element mesh
        tree: a BVH tree that stores the triangles of field's finite-element mesh
    """
    return jl.FEval.evaluate(pointsx,pointsy,field,tree)

def update_tree(tree,rescale_factor):
    jl.FEval.update_tritree(tree,rescale_factor)
    return
