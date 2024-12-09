resampling and evaluating fields
--------------------------------

In ``cbeam``, field values are saved only over the mesh nodes. If you have a field from one mesh and want to make it work with another mesh, you must resample the field on the new mesh's points. This is done using the ``FEval`` submodule, as follows. ::

    from cbeam import FEval

    # resample takes: field, input mesh, output mesh
    resampled_field = FEval.resample(field, mesh, newmesh)

If you want to propagate this resampled field, you then need to decompose the field into the modal basis of the propagator, which uses ``newmesh``. ::

    from cbeam.propator import Propagator

    # initialize propagator however
    # and characterize() or load()
    prop = Propagator(...) 

    mode_vector = prop.make_mode_vector(resampled_field)

Take a look at :doc:`mmi` for an example.

We can also evaluate a ``field`` computed on a ``mesh`` at an *arbitrary* ``point = (x,y)``. ``cbeam`` accelerates the repeated evaluation of fields on the same mesh by storing the mesh's triangles in a bounding volume hierarchy (BVH) tree. For instance, to perform the field evaluation mentioned above, we can use ::

    val = FEval.evaluate(point,field,tree)

where the arguments are:

1. ``point``: a length 2 ``numpy`` array containing the :math:`(x,y)` coordinates of the point to evaluate the field at.
2. ``field``: the field to be evaluated, discretized as a length :math:`M` array where :math:`M` is the number of mesh points.
3. ``tree``: the BVHtree representing the finite element mesh that ``field`` is discretized over (more on this next).

To construct a BVH tree, we use ::

    tree = FEval.create_tree_from_mesh(mesh)

You can also resample onto :math:`N` points at once (e.g. the points of another mesh, which is what ``resample()`` does under the hood) by passing an :math:`N\times 2` ``point`` array, or resample onto a grid using ``FEval.evaluate_grid()``.