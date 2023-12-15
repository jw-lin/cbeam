basic usage 
===================

--------
overview
--------

``cbeam`` provides the following submodules.

* ``cbeam.waveguide`` : used to define waveguides.

* ``cbeam.propagator`` : used to define propagation parameters and run propagations.

The typical workflow is to first define a waveguide, and then pass it to a propagator object to characterize the waveguide and then propagate fields through it.

-----------------------
setting up a waveguide
-----------------------

Waveguides in ``cbeam`` are, for the most part, defined functionally. There are several pre-defined classes, and users may define their own, inheriting from the base ``Waveguide`` class as a starting point. More guidance on how to do this can be found on the :doc:`waveguide </waveguide>` page.

Below are some pre-defined waveguides:

* ``CircularStepIndexFiber`` : a straight, circular, step-index optical fiber.
* ``RectangularStepIndexFiber`` : a straight, step-index fiber with a rectangular core.
* ``Dicoupler`` : 2x2 directional coupler with circular channels.
* ``Tricoupler`` : a 3x3 directional coupler with circular channels, in equilateral triangle configuration.
* ``PhotonicLantern`` : a linearly-tapered photonic lantern, with circular cores, cladding, and jacket.

I plan on adding the following (but they don't exist yet!)

* ``MMI``: a multimode interferometer
* ``RectangularDicoupler`` : a 2x2 directional coupler using rectangular channels.

All ``Waveguide`` objects have several important capabilities. First, the ``Waveguide`` object is responsible for creating a finite-element mesh, which will be used to solve for the waveguide's modes. Second, if waveguide's cross section varies with distance along the propagation axis, the mesh must be deformed to follow this variation - this is handled automatically in many cases. Finally, ``Waveguide`` objects have built-in plotting functions.

.. plot::

    from cbeam import waveguide
    rcore = 10
    rclad = 30
    ncore = 1.445
    nclad = 1.44

    fiber = waveguide.CircularStepIndexFiber(rcore,rclad,ncore,nclad)

    fiber.plot_mesh()