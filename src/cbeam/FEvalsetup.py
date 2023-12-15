from juliacall import Pkg as jlPkg
from juliacall import Main as jl
import cbeam,os

def FEvalsetup():
    path = os.path.dirname(cbeam.__file__)
    jlPkg.activate(path+"/FEval")
    jlPkg.add("PythonCall")
    jlPkg.precompile()
