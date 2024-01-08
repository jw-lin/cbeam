from juliacall import Main as jl
from juliacall import Pkg as jlPkg
import numpy as np


jlPkg.activate("BVHtree") 
jl.seval("using BVHtree")

def create_tree(tripoints):
    return jl.BVHtree.construct_tritree(tripoints)

def query(point,tree):
    jl_idx = jl.BVHtree.query(point,tree)
    return jl_idx-1

def get_idxs_and_weights(new_points,tree):
    triidxs,interpweights = jl.BVHtree.get_interp_weights(new_points,tree)
    return np.array(triidxs)-1,np.array(interpweights)



