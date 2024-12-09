directional coupler
===================
.. contents::
    :local:

The simplest directional coupler is composed of two embedded single-mode channels, which are "squeezed" to some minimum core-to-core separation over the course of some "coupling length". Along the coupling length, power between the channels oscillates. If the overall waveguide is symmetric, the net power transfer between the cores can vary from 0 to 1, and will oscillate sinusoidally with the coupling length.

A wavefront originally confined to one channel of a symmetric dicoupler will evenly excite both the symmetric and antisymmetric modes along the coupling length . Due to the difference in effective index, the supermodes beat against each other. The net effect is an oscillation in power between the dicoupler channels. As the coupling length changes, the amount of phase beating changes, which in turn changes the power splitting ratio of the dicoupler.

waveguide setup
---------------

We will use the ``Dicoupler`` class to define a dicoupler. First, let's set our parameters.

.. plot::
    :context:
    :nofigs:

    ### symmetric dicoupler propagation parameters ###

    wl = 1.55                           # wavelength, um
    dmin = 10.                          # minimum center-to-center separation of single mode channels
    dmax = 60.                          # maximum center-to-center separation of single mode channels
    coupling_length = 5000.             # length of coupling region (where SM channels are close)
    bend_length = coupling_length/4.   # approximate length of channel bends

    rcore = 3.                      # core radius. we will simulate a symmetric dicoupler, so core radii of both channels are the same

    nclad = 1.444                   # cladding refractive index
    ncore = nclad + 8.8e-3          # SM core refractive index

    # mesh params #
    core_res = 20                    # no. of line segments to use to resolve the core-cladding interface(s)
    clad_mesh_size = 20.0               # mesh size (triangle side length) to use in the cladding region
    core_mesh_size = 1.0                # mesh size (triangle side length) to use inside the cores

    tag = "test_dicoupler"

Note that for the ``Dicoupler`` class, we specify a coupling length, not the overall waveguide length like we did with the ``PhotonicLantern``. The overall waveguide length is auto-computed (in this basic implementation, it's two times the coupling length) and can be accessed through ``Dicoupler.z_ex``. 
Let's make the dicoupler and inspect the channel paths. I've made a built-in plotting function for this.

.. plot::
    :context:

    from cbeam import waveguide

    dicoupler = waveguide.Dicoupler(rcore,rcore,ncore,ncore,dmax,dmin,nclad,coupling_length,bend_length,core_res,core_mesh_size=core_mesh_size,clad_mesh_size=clad_mesh_size)

    # here we could adjust the boundary refinement mesh parameters, e.g.
    # dicoupler.min_mesh_size = <minimum mesh element size>

    dicoupler.plot_paths()

Lets take a look at the mesh, especially how it transforms with :math:`z`:

.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(1,2)
    
    dicoupler.plot_mesh(z=0,ax=axs[0],alpha=0.1)
    dicoupler.plot_mesh(z=dicoupler.z_ex/2,ax=axs[1],alpha=0.1)
    plt.show() 

In order to improve the regularity of the triangles, the outer cladding boundary is allowed to deform as the two cores are brought closer together.

mode solving
------------

Next, I'll initialize the propagator, and plot a mode.

.. plot::
    :context: close-figs

    from cbeam import propagator 
    dc_prop = propagator.Propagator(wl,dicoupler,Nmax=2)

    neff,modes = dc_prop.solve_at(z=dicoupler.z_ex/2)
    dc_prop.plot_cfield(modes[1],mesh=dc_prop.mesh) 

This is the antisymmetric mode in the coupling region.

characterization
----------------

Next, let's characterize and look at the coupling coefficients. For reference, this takes around 10s on my laptop.

.. plot::
    :context: close-figs
    
    # comment/uncomment below as necessary
    dc_prop.characterize(save=True,tag=tag) 
    # dc_prop.load(tag)

    dc_prop.plot_coupling_coeffs()

We see two large spikes, corresponding to a shift in modal basis from the individual channel modes to the symmetric and antisymmetric modes of the coupling region.

propagation
-----------

Let's launch light into one end and look at how the mode powers change with :math:`z`.

.. plot::
    :context: close-figs

    u0 = [1,0]
    zs,us,uf = dc_prop.propagate(u0)
    dc_prop.plot_mode_powers(zs,us)

We see that the light, initially confined in one of the channels, couples evenly into both modes within the couplng region, and then splits.

varying the coupling length
-----------------------------

Suppose we wanted to look at how the splitting ratio changes with the coupling length. We can play a trick that allows us to reuse the above calculation without rerunning ``characterize``. The idea is to apply a transformation to the $z$ array, preserving monotonicity, to change the length of the waveguide. Below is an example.

.. plot::
    :context: close-figs

    # we will run 100 dicoupler simulations with different lenghts
    stretch_amounts = np.linspace(0,10000,100)

    u0 = [1,0]
    pwrs = []

    for i,stretch in enumerate(stretch_amounts):
        zs = np.copy(dc_prop.zs)
        zs[np.argmax(zs>=dicoupler.z_ex/2):] += stretch # stretch out the z array

        dc_prop.make_interp_funcs(zs) # remake the interpolation functions
        zs,us,uf = dc_prop.propagate(u0,zs[0],zs[-1]) # rerun the propagator
        
        pwr = np.power(np.abs(uf),2)
        pwrs.append(pwr)

    pwrs = np.array(pwrs)

    pred_period = 4735 ## predicted oscillation period, see next section for the formula ##

    zmax = stretch_amounts[np.argmax(pwrs[:,0])] # translating the sinusoid to match - not trying to match absolute phase (see next section)
    
    # plot predicted cos^2 dependence
    plt.plot(stretch_amounts,np.power(np.cos(np.pi/pred_period*(stretch_amounts-zmax)),2),color='k',ls='dashed',label="predicted") 

    plt.plot(stretch_amounts,pwrs.T[0],label="channel 1")
    plt.plot(stretch_amounts,pwrs.T[1],label="channel 2")
    plt.legend(loc='best',frameon=False)
    plt.xlabel("dicoupler length")
    plt.ylabel("power")
    plt.title("output of dicoupler channels vs. of coupling length")
    plt.show()

In the above plot, I also show a "predicted" power output as a function of length. The derivation can be found in fiber optics textbooks. For an ideal, symmetric dicoupler with light injected solely into channel 1, the output power in channel 1 is 

.. math::
    P_1 = \cos^2(\kappa L)

For a symmetric dicoupler with circular cores, the oscillation wavenumber :math:`\kappa` has an empirical approximation [1]:

.. math::
    \kappa \approx \dfrac{\pi V}{2 k n_{\rm clad} r^2} \exp\left[ -\left(c_0 +c1\frac{d}{a}+c_2\frac{d^2}{a^2} \right) \right]


for :math:`k` the free-space wavenumber, :math:`r` the single-mode channel core radius, :math:`d` the inter-core spacing along the coupling length, and :math:`n_{\rm clad}` the refractive indexing of the cladding. The fiber :math:`V` number is defined as 

.. math::
    V = kr\sqrt{n_{\rm core}^2-n_{\rm clad}^2}

where :math:`n_{\rm core}` is the refractive index of the core material. The empirical constants are defined through:

.. math::
    c_0 &= 5.2789 - 3.663 V + 0.3841 V^2
    
    c_1 &= -0.7769 + 1.2252 V - 0.0152 V^2
    
    c_2 &= −0.0175 − 0.0064 V + 0.0009V^2

The above empirical formula has a quoted accuracy of <1% for :math:`1.5\leq V \leq 2.5` and :math:`2\leq d/a \leq 4.5`. For our dicoupler parameters, which fall in this range, the predicted period is :math:`\sim 4735 \mu {\rm m}`.

**References**

[1] R. Tewari and K. Thyagarajan, "Analysis of tunable single-mode fiber directional couplers using simple and accurate relations," in Journal of Lightwave Technology, vol. 4, no. 4, pp. 386-390, April 1986, doi: 10.1109/JLT.1986.1074731