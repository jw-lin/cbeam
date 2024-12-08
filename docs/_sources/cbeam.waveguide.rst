cbeam.waveguide
---------------
.. contents::
    :local:
    :depth: 1

This page contains the documentation for the ``waveguide`` submodule, which functionally defines different kinds of waveguides, enables the creation of custom waveguides, and handles mesh generation for finite element analysis.

^^^^^^^^^^^^^^^^^^^
Waveguide class
^^^^^^^^^^^^^^^^^^^

.. autoclass:: cbeam.waveguide.Waveguide
    :members:
    :exclude-members: deriv_transform
    :member-order: groupwise
    
    .. automethod:: __init__


"""""""""""""""""""""""""""""""""""""""
classes that inherit from ``Waveguide``
"""""""""""""""""""""""""""""""""""""""

.. autoclass:: cbeam.waveguide.CircularStepIndexFiber

    .. automethod:: __init__

.. autoclass:: cbeam.waveguide.RectangularStepIndexFiber

    .. automethod:: __init__

.. autoclass:: cbeam.waveguide.Dicoupler
    :members:

    .. automethod:: __init__

.. autoclass:: cbeam.waveguide.Tricoupler
    :members:

    .. automethod:: __init__

.. autoclass:: cbeam.waveguide.PhotonicLantern

    .. automethod:: __init__

.. autoclass:: cbeam.waveguide.TestPhotonicLantern

^^^^^^^^^^^^^^^^^
Prim3D class
^^^^^^^^^^^^^^^^^

.. autoclass:: cbeam.waveguide.Prim3D
    :members:
    :member-order: groupwise

    .. automethod:: __init__

"""""""""""""""""""""""""""""""""""""""
classes that inherit from ``Prim3D``
"""""""""""""""""""""""""""""""""""""""

.. autoclass:: cbeam.waveguide.Pipe

    .. automethod:: __init__

.. autoclass:: cbeam.waveguide.BoxPipe

    .. automethod:: __init__

^^^^^^^^^^^^^^^^
Prim2D class
^^^^^^^^^^^^^^^^

.. autoclass:: cbeam.waveguide.Prim2D
    :members:

    .. automethod:: __init__

"""""""""""""""""""""""""""""""""""""""
classes that inherit from ``Prim2D``
"""""""""""""""""""""""""""""""""""""""

.. autoclass:: cbeam.waveguide.Circle

    .. automethod:: make_points

.. autoclass:: cbeam.waveguide.Rectangle

    .. automethod:: make_points

.. autoclass:: cbeam.waveguide.Prim2DUnion

    .. automethod:: __init__
    .. automethod:: make_points