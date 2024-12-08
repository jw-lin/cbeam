quickstart
===================
.. contents::
    :local:
    :depth: 1
    
-----------
1. overview
-----------

``cbeam`` provides the following submodules.

* ``cbeam.waveguide`` : used to define waveguides.

* ``cbeam.propagator`` : used to define propagation parameters and run propagations.

The typical workflow is to first define a waveguide, and then pass it to a propagator object to characterize the waveguide and then propagate fields through it.

-------------------
2. waveguide basics
-------------------

Waveguides in ``cbeam`` are, for the most part, defined functionally. There are several pre-defined classes, and users may define their own, inheriting from the base ``Waveguide`` class as a starting point. The full documentation is :doc:`here </cbeam.waveguide>`.

Below are some pre-defined waveguides.

* ``CircularStepIndexFiber`` : a straight, circular, step-index optical fiber.
* ``RectangularStepIndexFiber`` : a straight, step-index fiber with a rectangular core.
* ``Dicoupler`` : 2x2 directional coupler with circular channels.
* ``Tricoupler`` : a 3x3 directional coupler with circular channels, in equilateral triangle configuration.
* ``PhotonicLantern`` : a linearly-tapered photonic lantern, with circular cores, cladding, and jacket.

I plan on adding the following (but they don't exist yet!)

* ``MMI``: a multimode interferometer
* ``RectangularDicoupler`` : a 2x2 directional coupler using rectangular channels.

``Waveguide`` objects have several important functions. *First*, the ``Waveguide`` object is responsible for creating a finite-element mesh, e.g. through ::

    mesh = Waveguide.make_mesh()

This mesh is used to solve for eigenmodes. *Second*, if waveguide's cross section varies with distance along the propagation axis, the mesh must be deformed to follow this variation; this is covered more in :ref:`wvg`. *Third*, ``Waveguide`` objects also have built-in plotting functions to view the mesh and refractive index profile, e.g. through ::

    Waveguide.plot_mesh(mesh=mesh)

So, for instance, you can run the following:

.. plot::

    from cbeam import waveguide

    # make a waveguide for testing - 6 port photonic lantern
    wvg = waveguide.TestPhotonicLantern()
    mesh = wvg.make_mesh()
    wvg.plot_mesh(mesh=mesh) #can also leave out mesh; a mesh will be auto-generated

---------------------
3. propagator basics
---------------------

The ``Propagator`` class is responsible for computing the necessary properties of the waveguide for coupled-mode propagation, as well as the propagation itself. The full documentation is  :doc:`here </reference>`.

A ``Propagator`` is initialized as

    prop = propagator.Propagator(wl,wvg,Nmax)

where the non-optional arguments are as follows.

1.  ``wl`` : the propagation wavelength (in the same spatial unit that the waveguide geometry is defined in)
2. ``wvg`` : a ``Waveguide`` object
3. ``Nmax`` : the number of propagating modes you want to track within the waveguide (only the highest effective index modes are tracked.)

^^^^^^^^^^^^^^^^^^^^^^^^^^
waveguide characterization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before we can perform optical propagations, we need the following as a function of :math:`z` (the longitudinal coordinate of the waveguide) :

1. the instanteous eigenmodes
2. the instanteous eigenvalues
3. the coupling coefficient matrix 

To compute these values, we use ::         

    Propagator.characterize(save,tag)

The results are saved to file when ``save=True`` (``tag`` is a string identifier of your choice). You can choose to characterize only a portion of the waveguide as well. Results are loaded through ::

    Propagator.load(tag)

For more details on capabilities, and some tips for tackling more complicated waveguides check out :doc:`advanced usage <advancedusage>`.

^^^^^^^^^^^
propagation
^^^^^^^^^^^

Once the waveguide has been characterized, we can propagate fields through it. The general syntax is ::

    zs,us,uf = prop.propagate(u0,zi,zf)

where ``u0`` the launch field, expressed in the modal basis of waveguide modes. You can propagate in the reverse direction if you set ``zi`` :math:`>` ``zf``. This function returns 3 items:

1. ``zs`` : an array of :math:`z` values selected by the diff eq solver used to solve the coupled-mode equations.
2. ``us`` : an array of amplitudes for the eigenmodes at each :math:`z` (with most of the complex phase oscillation factored *out*, as per coupled-mode theory).
3. ``uf`` : the final mode amplitudes (with phase oscillation factored *in*) - these are the actual complex-valued mode amplitudes at ``zf``, evaluated in the basis of the final eigenmodes. 

To convert a mode amplitude vector to a proper electric field, you can use ::

    field = Propagator.make_field(mode_vector,z)

where ``mode_vector`` is an array of complex-valued mode amplitudes (e.g. any column of ``u``, *but not* ``uf`` under default behavior) and ``z`` is the :math:`z` coordinate corresponding to the mode vector. You can also generate a plot with ``plot=True``. Otherwise, use the following for complex-valued fields ::
    
    Propagator.plot_cfield(field,z)

^^^^^^^^^^^^^^^^^^^^^^^
putting it all together
^^^^^^^^^^^^^^^^^^^^^^^
... looks something like this: 

.. plot::

    from cbeam.propagator import Propagator
    from cbeam.waveguide import TestPhotonicLantern

    # make the waveguide
    wvg = TestPhotonicLantern()

    wavelength = 1.55 # um
    num_modes = 6 # assuming we're using the 6-port lantern from earlier
    tag = "test"

    # make the propagator
    prop = Propagator(wavelength,wvg,num_modes)

    # characterization - uncomment below if you haven't run this yet
    # prop.characterize(save=True,tag=tag)
    
    # just use the the following if you already ran the above
    prop.load(tag)

    # propagation

    u0 = [1,0,0,0,0,0] # starting mode vector, corresponding to fundamental mode

    zs,us,uf = prop.propagate(u0) # default behavior is to propagate through the entire waveguide

    # get the fields and plot

    input_field = prop.make_field(u0,0)
    output_field = prop.make_field(us[:,-1],zs[-1])

    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(1,2,)

    prop.plot_cfield(input_field,z=0,ax=axs[0],show_mesh=True)
    prop.plot_cfield(output_field,z=zs[-1],ax=axs[1],show_mesh=True)

    # plot decoration
    axs[0].set_title("initial field")
    axs[1].set_title("final field")
    axs[1].set_xlim(-100,100)
    axs[1].set_ylim(-100,100)
    
    # must manually call show() if you're plotting on a premade axis
    plt.show()

Now you are ready to look at the examples!