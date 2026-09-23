testing
=======
.. contents::
    :local:
    :depth: 1

``cbeam`` ships a test-suite under ``tests/`` that is meant to catch
regressions as the package evolves. It has two layers:

* **unit tests** (``tests/unit``) - fast, isolated checks of every class and
  helper function in ``cbeam.waveguide``, ``cbeam.propagator`` and
  ``cbeam.FEval``.
* **integration tests** (``tests/integration``) - end-to-end runs, each one a
  rewrite of an example from this documentation (:doc:`basicusage`,
  :doc:`fib`, :doc:`PL`, :doc:`dc`, :doc:`mmi`, :doc:`PL19`). These build a
  waveguide, characterize it, propagate a field through it, and check physical
  invariants (power conservation, mode ordering, matrix (anti)symmetry) rather
  than exact numbers, which depend on the ``Gmsh`` mesh.

-------------------
1. what is covered
-------------------

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - file
      - covers
    * - ``unit/test_misc.py``
      - ``get_19port_positions``, ``blend``, ``dist``, ``rotate``, ``linear_taper``
    * - ``unit/test_prim2d.py``
      - ``Prim2D``, ``Circle``, ``Rectangle``, ``Prim2DUnion``
    * - ``unit/test_prim3d.py``
      - ``Prim3D``, ``Pipe``, ``LinearPipe``, ``Box``, ``BoxPipe``
    * - ``unit/test_waveguide.py``
      - the base ``Waveguide`` class: mesh generation, the ``z`` transform,
        ``transform_mesh``, IOR dictionaries, plotting
    * - ``unit/test_waveguide_classes.py``
      - every pre-defined waveguide (``CircularStepIndexFiber``,
        ``RectangularStepIndexFiber``, ``PhotonicLantern``,
        ``TestPhotonicLantern``, ``Dicoupler``, ``Tricoupler``,
        ``PlanarTricoupler``, ``OAMPhotonicLantern``)
    * - ``unit/test_feval.py``
      - ``cbeam.FEval``: BVH tree build / query / evaluate / grid / resample /
        ``transverse_gradient`` (a linear field is recovered exactly on a
        quadratic mesh)
    * - ``unit/test_propagator.py``
      - ``Propagator`` / ``ChainPropagator`` helpers and the full
        ``z``-invariant path (``solve_at`` -> ``characterize`` -> ``propagate``
        -> ``make_field`` -> save / load)
    * - ``integration/test_quickstart.py``
      - :doc:`basicusage`
    * - ``integration/test_fib.py``
      - :doc:`fib`
    * - ``integration/test_photonic_lantern.py``
      - :doc:`PL`
    * - ``integration/test_dicoupler.py``
      - :doc:`dc`
    * - ``integration/test_mmi.py``
      - :doc:`mmi`
    * - ``integration/test_pl19.py``
      - :doc:`PL19`

------------------
2. how to run them
------------------

You need the full runtime stack (see :doc:`installation`) plus ``pytest``,
and a working ``cbeam`` install (an editable install, ``pip install -e .``,
is fine). From the repository root::

    pytest -m "not slow"     # unit + light integration, ~30 s
    pytest                    # everything, ~15-20 min
    pytest -m slow            # only the full characterizations / propagations

The ``slow`` marker is on any test that runs a full adaptive
``Propagator.characterize`` / ``compute_modes`` / ``compute_neffs`` on a
tapering waveguide. The slow integration tests deliberately shorten the
devices from the documentation (for example a short leading slice of the
photonic lantern instead of the full 40 mm) - the numerical code path is
identical, only the number of ``z``-steps changes.

A few notes:

* ``pyproject.toml`` sets ``pythonpath = ["src"]``, so the suite always
  exercises the code in the checkout, not any other installed copy.
* ``tests/conftest.py`` forces the ``Agg`` matplotlib backend and, on the
  first run, instantiates the small Julia project in ``src/cbeam/FEval``
  (this is the same thing ``cbeam.FEvalsetup()`` does).
* Nothing is written to ``./data``; each ``Propagator`` in the suite gets a
  temporary ``save_dir``.

backends
~~~~~~~~

``cbeam.backend`` reads ``CBEAM_BACKEND`` once, at import time, so the two
backends are exercised by running the suite twice::

    CBEAM_BACKEND=numpy pytest        # the default
    CBEAM_BACKEND=jax   pytest        # requires jax + diffrax

Waveguide geometry and ``Gmsh`` meshing always run on host NumPy regardless of
``CBEAM_BACKEND``; the unit tests that only touch that code carry
``@pytest.mark.numpy_only`` and are skipped on the jax run. Everything that
solves modes, characterizes or propagates is run on both. The jax propagation
path uses a different ODE integrator (``diffrax`` ``Dopri5`` vs SciPy
``RK45``); its results agree with the NumPy path to ``~1e-8`` and the golden
check (below) widens its tolerances accordingly. Note the jax ``propagate()``
returns only the final mode vector, not the whole ``z`` trajectory.

golden numerical-regression layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On top of the physical-invariant assertions, each slow integration test pins
its stable numeric outputs (effective-index scans, sorted channel/mode power
spectra) against a committed reference set in ``tests/integration/_golden/``::

    CBEAM_GOLDEN=check pytest    # the default: compare, ~1e-9 tol on numpy
    CBEAM_GOLDEN=record pytest   # rewrite the reference set (deliberate changes only)
    CBEAM_GOLDEN=off pytest      # skip the comparison entirely

A missing reference file skips (does not fail) the comparison for that test.
See ``tests/integration/_golden/README.md`` for provenance and per-quantity
tolerances.

--------------------------------
3. a typical successful run
--------------------------------

Run in a dedicated Conda environment (Python 3.12, ``juliacall`` 0.9.35,
``cbeam`` installed editable from the checkout)::

    $ pytest -m "not slow"
    ........................................................................ [ 58%]
    ....................................................                     [100%]
    124 passed, 10 deselected in 31.93s

    $ pytest -m slow
    PASSED tests/integration/test_dicoupler.py::test_characterize_and_propagate
    PASSED tests/integration/test_dicoupler.py::test_vary_coupling_length_via_z_rescale
    PASSED tests/integration/test_fib.py::test_tapered_box_fiber_compute_neffs
    PASSED tests/integration/test_photonic_lantern.py::test_solve_at_midpoint
    PASSED tests/integration/test_photonic_lantern.py::test_characterize_propagate_and_channel_powers
    PASSED tests/integration/test_photonic_lantern.py::test_degen_groups_run
    PASSED tests/integration/test_pl19.py::test_19port_positions_hex_layout
    PASSED tests/integration/test_pl19.py::test_compute_neffs_scan
    PASSED tests/integration/test_pl19.py::test_chain_propagator_end_to_end
    PASSED tests/integration/test_quickstart.py::test_putting_it_all_together
    10 passed, 124 deselected in 929.78s (0:15:29)

Total: **134 passed, 0 failed**.
