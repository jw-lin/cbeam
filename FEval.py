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

def evaluate_func(field,tree):
    """ return a (julia) function of the point [x,y] corresponding to a given FE field """
    return jl.FEval.evaluate_func(field,tree)

def diff_func(field1,tree1,field2,tree2):
    """ return a (julia) function of the point [x,y] corresponding to the difference field2-field1 """
    return jl.FEval.diff_func(field1,tree1,field2,tree2)

def avg_func(field1,tree1,field2,tree2):
    return jl.FEval.avg_func(field1,tree1,field2,tree2)

def compute_func_over(func,xa,ya):
    return np.array(jl.FEval.evaluate(func,xa,ya))

def compute_coupling_coeff(field1,tree1,field2,tree2,xmin,xmax,tol=1e-6):
    """ compute the coupling coefficient matrix corresponding to field1 and field2 """
    _diff = diff_func(field1,tree1,field2,tree2)
    _avg = avg_func(field1,tree1,field2,tree2)
 
    return jl.FEval.inner_product(_avg,_diff,xmin,xmax,tol)

def compute_coupling_coeff2(field1,tree1,field2,tree2):
    """ compute the coupling coefficient matrix corresponding to field1 and field2 """

    return jl.FEval.compute_coupling_pcube(field1,tree1,field2,tree2)

def compute_coupling_pcube(field1,tree1,field2,tree2):
    """ compute the coupling coefficient matrix corresponding to field1 and field2. pcubature method. """

    return jl.FEval.compute_coupling_pcube(field1,tree1,field2,tree2)

def compute_coupling_simplex(field1,tree1,field2,tree2):
    """ compute the coupling coefficient matrix corresponding to field1 and field2. simplex method. """

    return jl.FEval.compute_coupling_simplex(field1,tree1,field2,tree2)

def FE_dot(field1,tree1,field2,tree2):
    return jl.FEval.FE_dot(field1,tree1,field2,tree2)

def compute_coupling_pert(field,tree1,IORvals,IORidxbounds1,IORidxbounds2,tree2):
    return jl.FEval.compute_coupling_pert(field,tree1,IORvals,IORidxbounds1,IORidxbounds2,tree2)

def compute_cob(field1,tree1,field2,tree2):
    return jl.FEval.compute_cob(field1,tree1,field2,tree2)