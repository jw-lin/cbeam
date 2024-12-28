cbeam.propagator
----------------

This page contains the documentation for the ``propagator`` submodule, which handles all things related to characterizing a ``Waveguide`` and propagating light through it. 

^^^^^^^^^^^^^^^^^^^^^
Propagator class
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cbeam.propagator.Propagator
    :members:
    :exclude-members: generate_isect_mesh,update_degen_groups,prop_setup
    :member-order: groupwise

    .. automethod:: __init__

^^^^^^^^^^^^^^^^^^^^^
ChainPropagator class
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cbeam.propagator.ChainPropagator
    :members:

    .. automethod:: __init__