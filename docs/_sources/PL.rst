photonic lantern
=================
.. contents::
    :local:
    :depth: 2

The photonic lantern is a tapered waveguide that looks like a normal step-index optical fiber on one end, and a multicore fiber (or similarly, a bundle of single-mode fibers) at the other end. In this section we'll use ``cbeam`` to simulate the propagation through a 6-port photonic lantern. 

First, let's construct the lantern, using the ``PhotonicLantern`` class:

.. plot::
    :context:
    :nofigs:

    from cbeam.waveguide import PhotonicLantern
    import numpy as np

    wl = 1.55                       # wavelength, um
    taper_factor = 8.               # relative scale factor between frontside and backside waveguide geometry    
    rcore = 2.2/taper_factor        # radius of tapered-down single-mode cores at frontside (small end), um
    rclad = 10                      # radius of cladding-jacket interface at frontside, um
    rjack = 30                      # radius of outer jacket boundary at frontside, um
    z_ex = 40000                    # lantern length, um

    nclad = 1.444                   # cladding refractive index
    ncore = nclad + 8.8e-3          # SM core refractive index
    njack = nclad - 5.5e-3          # jacket (low-index capillary) refractive index

    t = 2*np.pi/5                   
    core_offset = rclad*2/3         # offset of outer ring of cores from center

    # initial core positions. the first core is at (0,0) and the other 5 form a pentagon at a distance <core_offset> from the center
    core_pos = np.array([[0,0]] + [[core_offset*np.cos(i*t),core_offset*np.sin(i*t)] for i in range(5)])

    # we'll leave the optional args at their defaults
    # below is actually equivalent to TestPhotonicLantern
    lant = PhotonicLantern(core_pos,[rcore]*6,rclad,rjack,[ncore]*6,nclad,njack,z_ex,taper_factor)


solving at :math:`z`
--------------------

We can do mode solves at specific :math:`z` values as before. Below I plot eigenmode 0 at the halfway point of the lantern.

.. plot::
    :context:
    
    from cbeam.propagator import Propagator
    wavelength = 1.55 # um
    PLprop = Propagator(wavelength,lant,6) # 6 modes

    PLneff,PLmodes = PLprop.solve_at(z=z_ex/2.)
    PLprop.plot_cfield(PLmodes[0],z=z_ex/2.)

solving through the waveguide
-----------------------------

We can also characterize the waveguide to get the effective indices and modes as a function of :math:`z`. For reference, this takes around a minute on my laptop. Below I also plot the effective indices.

.. plot::
    :context: close-figs

    # comment/uncomment below as necessary
    PLprop.characterize(save=True,tag="test")

    # PLprop.load("test")

    PLprop.plot_neffs()

We see that the eigenmodes are initially divided into 4 groups by effective index. Upon closer inspection, we could determine that modes 1&2 are degenerate, as well as 3&4. 

coupling coefficients
---------------------

Let's also plot the coupling coefficients.

.. plot::
    :context: close-figs

    PLprop.plot_coupling_coeffs()

The vertical lines indicate the :math:`z` values which ``cbeam`` adaptively chose to compute at. The plot also looks kind of spiky! But the spikes are not particularly surprising since we have a lot of mode degeneracy, so our eigenbasis can rotate more or less freely with :math:`z`. This is fine as long as the spikes sufficiently well-sampled in :math:`z`.


dealing with degenerate modes
-----------------------------

While the above calculation is fine, ``cbeam`` provides a way to "derotate" a degenerate eigenbasis, which can improve computation speed and accuracy. This involves specifying which modes are degenerate in ``Propagator.degen_groups``:

.. plot::
    :context: close-figs
    :nofigs:

    # modes 1&2 , 3&4 are degenerate
    PLprop.degen_groups = [[1,2],[3,4]]

I will run a ``characterize()`` again to show how the coupling coefficients change, even though the waveguide and the physics are the same.

.. plot::
    :context: close-figs

    # comment/uncomment below as necessary
    PLprop.characterize(save=True,tag="testdegen")
    # PLprop.load("testdegen")
    
    PLprop.plot_coupling_coeffs()

Comparing this plot with the previous, we notics that coupling coefficients are lower now, as expected.


extract channel powers
----------------------

Degeneracy throws one last complication at us. Suppose we want to propagate a field through the lantern and get the output powers in each single-mode core. A simple propagation will not give us the information, because the modes at the end of the waveguide might not match the modes we are interested in. I will propagate the fundamental mode through the lantern to illustrate.

.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt
    import numpy as np

    u0 = [1,0,0,0,0,0] # launch field, LP01 
    # you can try loading "test" here as well
    zs,us,uf = PLprop.propagate(u0)

    PLprop.plot_mode_powers(zs,us)
    plt.show()

5 of the modes should share the same power, which we clearly do not see. To properly extract the channel powers, we can use ``Propagator.to_channel_basis(uf)``. 

.. plot::
    :context: close-figs
    :nofigs:

    # note that this function uses the phased mode amplitudes uf, not the unphased amplitudes in u.
    amps = PLprop.to_channel_basis(uf)
    
    print(np.power(np.abs(amps),2))

This should give the following channel powers (or close to it):

.. testoutput::

    [0.49554017 0.10090713 0.10091332 0.10088415 0.10088725 0.10087432]