import os
import juliacall
import cbeam

# Use Julia's string evaluation to load Pkg into the Julia runtime environment
juliacall.Main.seval("using Pkg")
jlPkg = juliacall.Main.Pkg

def FEvalsetup():
    path = os.path.dirname(cbeam.__file__)
    jlPkg.activate(path + "/FEval")
    jlPkg.resolve()
    jlPkg.precompile()
