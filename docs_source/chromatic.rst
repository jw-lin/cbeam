chromatic response
==================
.. contents::
    :local:

A ``Propagator`` is built for a single wavelength: its eigenmodes, effective indices and coupling coefficients all depend on it. To see how a device behaves across a band, we therefore characterize it once per wavelength. The ``chromatic`` submodule runs that loop for any waveguide. We describe the device at one wavelength in a function ``build(wl)`` that returns a characterized propagator, and ``chromatic.sweep()`` calls it at every wavelength, evaluates some quantities on the result (by default the transfer matrix and the effective indices at both ends) and stacks them along a wavelength axis.

As an example, let's look at how the splitting ratio of the directional coupler from the :doc:`directional coupler <dc>` example changes with wavelength.

the build function
------------------

We reuse the coupler parameters from that example.

.. plot::
    :context: close-figs
    :nofigs:

    import numpy as np
    from cbeam import waveguide, propagator, chromatic

    wl_ref = 1.55                       # design wavelength, um
    dmin = 10.                          # minimum center-to-center separation of single mode channels
    dmax = 60.                          # maximum center-to-center separation of single mode channels
    coupling_length = 5000.             # length of coupling region (where SM channels are close)
    bend_length = coupling_length/4.    # approximate length of channel bends
    rcore = 3.                          # core radius
    nclad = 1.444                       # cladding refractive index at wl_ref
    ncore = nclad + 8.8e-3              # core refractive index at wl_ref

    # mesh params #
    core_res = 15
    clad_mesh_size = 20.0
    core_mesh_size = 1.0

``build(wl)`` makes the waveguide, sets up a ``Propagator`` at ``wl`` and characterizes it. ``chromatic.load_or_characterize()`` loads a saved characterization if there is one, and otherwise characterizes and saves; ``chromatic.wavelength_tag()`` gives each wavelength its own tag. The tag is all that ``Propagator.load()`` checks, so it should also name anything else the characterization depends on -- here, whether the indices are dispersive.

.. plot::
    :context: close-figs
    :nofigs:

    def build(wl, dispersive=False):
        if dispersive:
            ncl, nco = chromatic.dispersive_index([nclad, ncore], wl, wl_ref)
        else:
            ncl, nco = nclad, ncore
        dc = waveguide.Dicoupler(rcore, rcore, nco, nco, dmax, dmin, ncl, coupling_length,
                                 bend_length, core_res, core_mesh_size=core_mesh_size,
                                 clad_mesh_size=clad_mesh_size)
        prop = propagator.Propagator(wl, dc, Nmax=2)
        prop.z_acc = -1     # loosen accuracy, as in the coupler example
        tag = "dc_silica" if dispersive else "dc_fixed"
        chromatic.load_or_characterize(prop, chromatic.wavelength_tag(tag, wl))
        return prop

Anything that depends on wavelength goes in ``build``: the mesh, the mode bookkeeping (``degen_groups``, ``skipped_modes``), or the refractive indices.

the sweep
---------

Each characterization takes around 10 seconds, so 21 wavelengths take a few minutes the first time; afterwards they are loaded from disk.

.. plot::
    :context: close-figs
    :nofigs:

    wls = np.linspace(1.45, 1.65, 21)
    res = chromatic.sweep(build, wls)

``res["transfer_matrix"]`` holds one :math:`2\times2` transfer matrix per wavelength, in the channel basis (column :math:`j` is the output for light launched in channel :math:`j`). The coupler is lossless, so each column's powers add up to 1, and the cross-coupled power :math:`|T_{10}|^2` is the splitting ratio.

.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt

    power = np.abs(res["transfer_matrix"])**2       # (n_wl, n_out, n_in)
    plt.plot(wls, power[:, 1, 0], label="cross")
    plt.plot(wls, power[:, 0, 0], label="bar")
    plt.xlabel(r"wavelength ($\mu$m)")
    plt.ylabel("output power fraction")
    plt.legend()
    plt.show()

The coupling strength between the two cores grows with wavelength (the modes spread further into the cladding), so the power exchanged along the coupling length is far from constant over the band.

chromatic refractive indices
----------------------------

So far the indices have been the same at every wavelength. ``chromatic.dispersive_index()`` gives a simple chromatic model for doped silica: every index follows the dispersion of fused silica (``chromatic.silica_index()``, a Sellmeier fit), while the index contrasts stay those of the design wavelength. The ``build`` function above already supports it.

.. plot::
    :context: close-figs

    res_si = chromatic.sweep(lambda wl: build(wl, dispersive=True), wls)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(wls, res["neffs_in"][:, 0], label="fixed indices")
    axs[0].plot(wls, res_si["neffs_in"][:, 0], label="silica dispersion")
    axs[0].set_ylabel(r"$n_{\rm eff}$ at the input")
    axs[1].plot(wls, (np.abs(res["transfer_matrix"])**2)[:, 1, 0], label="fixed indices")
    axs[1].plot(wls, (np.abs(res_si["transfer_matrix"])**2)[:, 1, 0], label="silica dispersion")
    axs[1].set_ylabel("cross-coupled power")
    for ax in axs:
        ax.set_xlabel(r"wavelength ($\mu$m)")
        ax.legend()
    plt.tight_layout()
    plt.show()

The material dispersion moves every effective index by about the same amount, so here it barely changes the coupling: in this weakly guiding coupler the splitting depends on the index contrast, which the model keeps fixed.

other waveguides and measures
-----------------------------

Nothing above is specific to the coupler. ``build`` can return any characterized ``Propagator``, or a ``ChainPropagator`` -- e.g. a :doc:`photonic lantern <PL19>` characterized in two segments, with the second segment's initial conditions taken from the first:

.. code-block:: python

    def build(wl):
        lantern = waveguide.PhotonicLantern(...)          # indices may depend on wl
        front = propagator.Propagator(wl, lantern, Nmax)
        back = propagator.Propagator(wl, lantern, Nmax)
        # ... front / back degen_groups and skipped_modes ...
        chromatic.load_or_characterize(front, chromatic.wavelength_tag("pl_front", wl), 0, z_split)
        back.load_init_conds(front)
        chromatic.load_or_characterize(back, chromatic.wavelength_tag("pl_back", wl), z_split, lantern.z_ex)
        return propagator.ChainPropagator([front, back])

Other quantities can be collected with ``measures``, a dict of functions of the propagator; each must return an array of the same shape at every wavelength. For instance, the output channel powers for one particular input:

.. code-block:: python

    u0 = ...    # input mode amplitudes
    def channel_powers(prop):
        zs, us, uf = prop.propagate(u0)
        return np.abs(prop.to_channel_basis(uf))**2

    res = chromatic.sweep(build, wls, measures={"powers": channel_powers})

Only one wavelength's propagator is kept in memory at a time, which matters for large waveguides.
