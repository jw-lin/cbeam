propagator settings
-------------------
.. contents::
    :local:
    :depth: 2

:math:`z`-stepping parameters for characterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Propagator.characterize()`` applies an adaptive algorithm to select the :math:`z` step size between calculations. This algorithm is as follows. At a given step, we interpolate through the eigenmodes at the last few :math:`z` values and compare the current eigenmodes with the extrapolated prediction. If the error is below the threshhold ``zstep_tol``, then the step is accepted, and if the error is less than a tenth the threshhold, the next step is doubled; otherwise, the step is divided in two and the couping-coefficient matrix is re-computed at this new step. The starting step size is 10. ``zstep_tol`` and other parameters are accessible as ``Propagator`` class attributes, listed below:

* ``zstep_tol`` : this tolerance parameter controls the adaptive stepping. Smaller = more careful stepping (default 1e-3).

* ``fixed_zstep`` : set this to a numeric value to use a fixed :math:`z` step, bypassing the adaptive scheme.

* ``min_zstep`` : this is the minimum :math:`z` step that can be chosen by the adaptive scheme, default 1.25.

* ``max_zstep`` : this is the maximum :math:`z` step that can be chosen by the adaptive scheme. This is useful to prevent the adaptive stepper from skippin over small, peaked features in the cross-coupling matrix, default 640.

specifying degenerate modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The presence of degenerate modes in a waveguide will typically slow down computation and can sometimes cause issues with numerical stability. This is because such modes, as computed by the finite-element solver, can "rotate" rapidly as we advance in :math:`z`. This produces sharp peaks in the coupling-coefficients. 

If we know that certain modes will remain degenerate throughout the entire waveguide (e.g. we ran ``Propagator.compute_neffs()`` before hand), we can pass that information into the following ``Propagator`` attribute.

* ``degen_groups`` : a nested list of indices which identify groups of modes that remain degenerate throughout the entire waveguide. For instance, if we know *a priori* that modes (1,2) and modes (3,4) form degenerate pairs, we could pass in ``[[1,2],[3,4]]``. **This is highly recommended whenever possible.**

Given this information, the code will apply a change of basis to each degenerate mode group which minimizes rotation with :math:`z`. 

What if the mode degeneracies are not consistent through the waveguide? As far as I can tell, this is pretty tricky to simulate, especially in cases where eigenvalues split and converge repeatedly. Thus, my official advice (for now) is to break up such a waveguide into regions in :math:`z` where the degeneracy structure is consistent, and perform the characterizations and propagations in parts.

accessing the data
^^^^^^^^^^^^^^^^^^^

After running ``characterize()``, you may want to take a closer look at how the eigenvalues, eigenmodes, and coupling coefficients change with :math:`z`. This data is returned by ``characterize()`` but is also stored in the following variables:

* ``Propagator.zs`` : the array of :math:`z` values traversed by ``characterize()``.

* ``Propagator.neffs`` : the effective refractive indices of the eigenmodes. The mode effective index :math:`n_e` is related to the eigenvalue :math:`w` by :math:`n_e = \sqrt{w/k^2}`, where :math:`k` is the free space wavenumber.

* ``Propagator.vs`` : the eigenmode basis computed over ``zs``.

* ``Propagator.cmats`` : the coupling coefficient matrix computed over ``zs``.

If ``save=True`` was set, the above arrays are also saved to file and can be loaded with ``Propagator.load()``.

This data may be plotted using the following functions.

* ``plot_neffs()`` : plot the effective indices.

* ``plot_waveguide_mode(i)`` : plot eigenmode ``i``; the plot comes with a slider for :math:`z`.

* ``plot_coupling_coeffs()`` : plot the coupling coefficients between the modes.

Interpolation functions for the above data are automatically generated, but you can remake these functions manually with 

* ``Propagator.make_interp_funcs(zs)``

which is useful if you want to transform or rescale the :math:`z` coordinate. These interpolation functions are at 

* ``Propagator.compute_neff(z)``

* ``Propagator.compute_v(z)``

* ``Propagator.compute_cmat(z)``

propagation parameters
^^^^^^^^^^^^^^^^^^^^^^

To solve the coupled-mode equations, ``cbeam`` uses ``scipy.integrate.solve_IVP()``. The integration scheme used by this function can be set through the class attribute

* ``Propagator.solver`` : the scheme (the "method" argument) to pass to ``solve_IVP()``; default is ``"RK45"`` (4th order adaptive Runge-Kutta). 

Whe propagating, there is also the option to include a minor WKB-like correction to the coupled-mode equations. Set this through the attribute

* ``Propagator.WKB`` (bool): this controls whether or not the correction is included in the coupled-mode equations (default ``False``).

The effect of this correction is typically negligible, and becomes important only if mode propagation constants change significantly through the waveguide.