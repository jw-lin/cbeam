from setuptools import setup
from setuptools.command.install import install
import os
from juliacall import Pkg as jlPkg
class PostInstallCommand(install):
    """Post-installation for installation mode. Install julia dependencies."""
    def run(self):
        install.run(self)
        os.chdir("./src/cbeam")
        jlPkg.activate("FEval")
        jlPkg.add("PythonCall")
        jlPkg.add("StaticArrays")
        jlPkg.add("GrundmannMoeller")
        jlPkg.add("Cubature")

setup(include_package_data=True,cmdclass={'install': PostInstallCommand,},)

