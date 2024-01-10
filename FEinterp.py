from juliacall import Main as jl
from juliacall import Pkg as jlPkg
import numpy as np

jlPkg.activate("FEinterp") 
jl.seval("using FEinterp")

def create_tree(tripoints):
    return jl.FEinterp.construct_tritree(tripoints)

def query(point,tree):
    jl_idx = jl.FEinterp.query(point,tree)
    return jl_idx-1

def get_idxs_and_weights(new_points,tree):
    triidxs,interpweights = jl.FEinterp.get_interp_weights(new_points,tree)
    return np.array(triidxs)-1,np.array(interpweights)



