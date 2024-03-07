from setuptools import setup
from setuptools.command.install import install
import os

class PostInstallCommand(install):
    """Post-installation for installation mode. Install julia dependencies."""
    def run(self):
        install.run(self)
        import cbeam
        path = os.path.dirname(cbeam.__file__)
        os.chdir(path)
        from juliacall import Pkg as jlPkg
        jlPkg.activate("FEval")
        jlPkg.add("PythonCall")
        jlPkg.add("StaticArrays")
        jlPkg.add("GrundmannMoeller")
        jlPkg.add("Cubature")

setup(include_package_data=True,cmdclass={'install': PostInstallCommand,},)

