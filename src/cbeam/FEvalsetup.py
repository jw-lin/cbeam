from juliacall import Main as jl
import cbeam,os

# older juliacall re-exported Pkg (``from juliacall import Pkg``); newer versions
# dropped it. access Julia's Pkg through Main instead, which works everywhere.
jl.seval("import Pkg")
jlPkg = jl.Pkg

def FEvalsetup():
    path = os.path.dirname(cbeam.__file__)
    jlPkg.activate(path+"/FEval")
    jlPkg.resolve()
    jlPkg.precompile()
