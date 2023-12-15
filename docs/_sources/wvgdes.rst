making custom waveguides
------------------------
.. contents::
    :local:
    :depth: 2


Defining a custom waveguide is kinda complicated without a proper CAD environment. In ``cbeam``, a ``Waveguide`` is a nested list of 3D objects, represented by the ``Prim3D`` class. In turn, a ``Prim3D`` is a 2D shape, like a circle or rectangle, combined with some functional :math:`z` dependence. This 2D shape is represented by the ``Prim2D`` class.

This structure was chosen to make it easier to construct waveguides: the tradeoff is that it is harder to add new 2D shapes, since these shapes need some extra machinery needs to inform the mesh transformation law. This section gives a bottom-up overview of this hierarchy.

the ``Prim2D`` class
^^^^^^^^^^^^^^^^^^^^^^^^

The most basic parent class that represents a refractive index geometry is a ``Prim2D``. Each ``Prim2D`` stores a refractive index value and an array of :math:`(x,y)` points bounding the physical region which contains that index value. A ``Prim2D`` is generically initialized through ::

    prim2D = waveguide.Prim2D(n,points)

where ``n`` is the refractive index inside the boundary and ``points`` is an :math:`N\times 2` array of :math:`(x,y)` corresponding to the boundary. The first and last point are automatically connected. ``points`` can also be ``None``, since the point array can be generated with functional dependence, as mentioned next.  

To make specific types of geometries, users are encouraged to define subclasses that inherit from ``Prim2D`` and generate ``points`` according to specific rules. These subclasses should implement their own ``make_points()`` functions, which at minimum should take in some set of arguments (like radius, for a circle primitive) and return the corresponding point array. Subclasses should also implement the following functions:

1. ``boundary_dist(x,y)`` : compute the minimum distance between the point :math:`(x,y)` and the primitive boundary, returning a negative value if the point is inside the boundary. 

2. ``nearest_boundary_point(x, y)`` : compute the closest boundary point to the point :math:`(x,y)`.

These functions enable the general mesh transformation law. See the ``Circle`` and ``Rectangle`` classes for examples. 

More complicated primitives can be created through the ``waveguide.Prim2DUnion`` class.

the ``Prim3D`` class
^^^^^^^^^^^^^^^^^^^^

The next level in complexity is the ``Prim3D`` class, which combines a ``Prim2D`` with rules that define evolution with respect to :math:`z`. A ``Prim3D`` is initialized as ::

    prim3D = waveguide.Prim2D(prim2D,label)

where ``prim2D`` is a ``Prim2D`` object representing the cross-section of the ``Prim3D`` and ``label`` is a user-specified string to attach to the bound region (e.g. "core" or "cladding" for a step-index fiber.)

Users can make their own ``Prim3D`` subclasses, which implement their desired :math:`z`-dependence. Inheriting classes must implement the function ``make_points_at_z()``, which updates ``prim2D`` to the desired :math:`z` coordinate and returns the point array. Two subclasses currently implemented are ``Pipe`` and ``BoxPipe``, which can be used to represent primitives with circular or rectangular cross-sections.

As mentioned in Section 1, ``cbeam`` models the :math:`z`-evolution of a waveguide by applying a continuous spatial transformation to a finite element mesh. A basic transformation that should work for a variety of different waveguide geometries is already implemented. However, to prevent this transformation from touching the mesh points within primitive boundaries, custom ``Prim3D`` subclasses should also implement the function ``transform_point_inside()``, which will override the mesh transformation rules within the primitive boundary. You can look at ``Pipe`` and ``BoxPipe`` as examples.


.. _wvg:

the ``Waveguide`` class
^^^^^^^^^^^^^^^^^^^^^^^

A ``Waveguide`` stores a (potentially nested) list of ``Prim3D`` objects, which we will call ``prim3Dgroups``. The refractive index profile of each top-level element in ``prim3Dgroups`` is overwritten by the immediate next element. Any element which is a list of ``Prim3D`` objects is treated as a single ``Prim3D``. :math:`^\dagger` So, an optical fiber might store its primitives as ``[cladding,core]``, where ``cladding`` and ``core`` are ``Pipe`` objects corresponding to the cladding and core regions; a multicore fiber could have the structure ``[cladding,[core1,core2,...]]`` and a trenched fiber could have ``[cladding,trench,core]``. A ``Waveguide`` is initialized as ::

    wvg = Waveguide(prim3Dgroups)

Users can define their own ``Waveguide`` subclasses, which initialize a set of ``Prim3D`` objects and arranges them as desired. A ``Waveguide``  can also implement the following function: 

* ``transform(x0,y0,z0,z)`` : this function takes a point :math:`(x_0,y_0,z_0)` and returns a new point :math:`(x,y)` at the longitudinal coordinate :math:`z`. This transformation is used give :math:`z` dependence to the mesh geometry in a continuous manner. 

This will override the base ``Waveguide.transform()`` function, which works for a variety of waveguides but may be slow. As an example, the ``PhotonicLantern`` class overrides ``transform()``.

The ``Waveguide`` class also handles mesh generation, which can be tuned via ``Waveguide`` class attributes. We will cover this next.

:math:`^\dagger` ``Prim3D`` objects in the same sublist should never intersect. However, they can have different ``label`` s and refractive indices. If you want to model intersecting primitives, you should make a ``Prim2DUnion``, and then apply some :math:`z`-dependence to make a custom ``Prim3D``.