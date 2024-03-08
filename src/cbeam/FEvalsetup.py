from juliacall import Pkg as jlPkg
from juliacall import Main as jl
import cbeam,os

def FEvalsetup():
    path = os.path.dirname(cbeam.__file__)
    #jlPkg.develop(jlPkg.PackageSpec(path = path+"\FEval") )
    jlPkg.activate(path+"/FEval")
    jlPkg.add("StaticArrays")
    jlPkg.add("PythonCall")
    jlPkg.add("GrundmannMoeller")
    jlPkg.add("Cubature")
    jlPkg.precompile()
