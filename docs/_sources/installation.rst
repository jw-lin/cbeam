installation
===================
.. contents::
    :local:
    :depth: 1

~~~~~~~~~~~~~~~~~~~~
1. dependencies
~~~~~~~~~~~~~~~~~~~~

You will need both a **Python3** and a `Julia <https://julialang.org/downloads/>`_ installation. Then, make sure you have the following packages:

**General**: ``Gmsh`` (download  `here <https://gmsh.info/>`_)

**Python**: ``numpy``, ``scipy``, ``juliacall``, ``wavesolve``, ``pygmsh``, ``meshio``, ``matplotlib`` (All ``pip`` installable by name besides ``wavesolve``, download `here <https://github.com/jw-lin/wavesolve>`_)

~~~~~~~~~~~~~~~~~~~~~
2. download ``cbeam``
~~~~~~~~~~~~~~~~~~~~~

Use pip: ::

    pip install git+https://github.com/jw-lin/cbeam.git

~~~~~~~~~~~~
3. setup
~~~~~~~~~~~~

To get the Julia dependencies and precompile the Julia code, run :: 

    import cbeam
    cbeam.FEvalsetup()

e.g. in an IPython session. You only need to do this once per installation. Now you should be able to run the :doc:`examples </examples>`.

~~~~~~~~~~~
4. updating
~~~~~~~~~~~

To update ``cbeam``, run ::

    pip install --upgrade git+https://github.com/jw-lin/cbeam.git

When updating, you should re-run ``cbeam.FEvalsetup()``.
