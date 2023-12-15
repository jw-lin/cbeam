making custom waveguides
------------------------

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

