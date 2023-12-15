meshing
-------

Mesh generation is handled through the ``Waveguide`` class function ``make_mesh()``. This function will adaptively refine the mesh element size near boundaries so that mesh sizes vary smoothly over the waveguide cross-section. The target mesh size is computed as follows. For a single ``Prim3D`` object, a target mesh size at a given :math:`(x,y)` point is computed as 

.. math::

    {\rm target \, size} = d_0\left(1+ \dfrac{s \, d(x,y)}{d_0} \right)^p

where :math:`d_0` is the "default" mesh size set by the resolution of the ``Prim3D`` boundary, :math:`d(x,y)` is the distance between the point :math:`(x,y)` and the primitive's boundary, and :math:`s` and :math:`p` are variables. 
These parameters are set through class attributes, covered below; higher values mean that mesh size will increase more rapidly away from the boundary. For multiple primitives, a target size is computed for each and the minimum size is taken. Then the target size is clipped between a minimum and maximum allowed value. This function is implemented in ::

    Waveguide.compute_mesh_size(x,y,s,p,min_size,max_size)

The parameter values which ``make_mesh()`` will use can be specified through the following ``Waveguide`` class attributes:

* ``mesh_dist_scale`` : :math:`s`, the mesh boundary refinement linear distance scaling. Default 0.5; unitless.

* ``mesh_dist_power`` : :math:`p`, mesh boundary refinement power scaling. Default 1; unitless.

* ``min_mesh_size`` : minimum allowed mesh size, default 0.1.

* ``max_mesh_size`` : maximum allowed mesh size, default 10.

.. note::

    **Mesh units and wavelength units should be consistent**.

Users can also specify a target mesh size, and toggle boundary refinement on a per-primitive basis. This is done through the following ``Prim3D`` attributes: 

* ``mesh_size`` : target mesh size within the boundary of the primitive (otherwise the mesh size is set by the scheme described above.)

* ``skip_refinement`` : whether or not mesh refinement should be applied at the primitive boundary. The outer boundary of the entire mesh should have this set to ``True`` (no refinement); default ``False``.

* ``preserve_shape`` : whether the mesh transformation is allowed to deform the boundary. Default ``True`` (not allowed), but the outer boundary of the mesh have this ``False``.