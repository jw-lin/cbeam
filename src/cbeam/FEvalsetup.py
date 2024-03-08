from juliacall import Pkg as jlPkg
import cbeam,os

def FEvalsetup():
    path = os.path.dirname(cbeam.__file__)
    jlPkg.activate(path+"/FEval")
    jlPkg.add("PythonCall")
    jlPkg.add("StaticArrays")
    jlPkg.add("GrundmannMoeller")
    jlPkg.add("Cubature")
