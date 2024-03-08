from juliacall import Pkg as jlPkg
from juliacall import Main as jl
import cbeam,os

def FEvalsetup():
    jlPkg.add("PythonCall")
    jl.seval("using PythonCall")
    path = os.path.dirname(cbeam.__file__)
    #jlPkg.develop(jlPkg.PackageSpec(path = path+"\FEval") )
    jlPkg.activate(path+"/FEval")
    jlPkg.add("StaticArrays")
    jlPkg.add("PythonCall")
    jlPkg.add("GrundmannMoeller")
    jlPkg.add("Cubature")
