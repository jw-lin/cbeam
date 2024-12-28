.. cbeam documentation master file, created by
   sphinx-quickstart on Wed Dec  4 16:47:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

----------------------------------------
cbeam: a coupled-mode optical propagator
----------------------------------------

``cbeam`` is a Python package which simulates the propagation of light through slowly varying waveguides. Under the hood, it uses coupled-mode theory combined with the method of finite elements, and is typically much faster than FD-BPM when applicable. If you're interested in the math, check out the arXiv paper `here <https://arxiv.org/abs/2411.08118>`_. 

If you have questions, suggestions, or would like to contribute, feel free to contact me at ``jon880@astro.ucla.edu``.

.. note::

   ``cbeam`` is still under active development. Currently, the package assumes weak guidance and unidirectional/paraxial propagation. Vectorial propagation, radiative losses, graded refractive indices, and back reflection are not supported (for now). There may be bugs.


^^^^^^^^^^^^^^^^^
contents
^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   installation
   basicusage
   examples
   advancedusage
   reference
