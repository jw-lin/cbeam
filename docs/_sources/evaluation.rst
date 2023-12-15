evaluating fields
-----------------

In ``cbeam``, field values are saved only over the mesh nodes. To evaluate a ``field`` computed on a ``mesh`` at an *arbitrary* ``point = (x,y)``, we use the ``FEval`` submodule. This package accelerates the evaluation of finite element fields by storing mesh triangles in a bounding volume hierarchy (BVH) tree. For instance, to perform the field evaluation mentioned above, we can use ::

    val = FEval.evaluate(point,field,tree)

where the arguments are:

1. ``point``: a length 2 ``numpy`` array containing the :math:`(x,y)` coordinates of the point to evaluate the field at.
2. ``field``: the field to be evaluated, discretized as a length :math:`M` array where :math:`M` is the number of mesh points.
3. ``tree``: the BVHtree representing the finite element mesh that ``field`` is discretized over (more on this next).

To construct a BVH tree, you can use ::

    tree = FEval.create_tree_from_mesh(mesh)

You can also resample onto :math:`N` points at once (e.g. the points of another mesh) by passing an :math:`N\times 2` ``point`` array, or resample onto a grid using ``FEval.evaluate_grid()``.