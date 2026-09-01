"""Integration test rewritten from ``docs_source/fib.rst`` (fiber mode solving)."""

import numpy as np
import pytest

from cbeam import waveguide
from cbeam.propagator import Propagator

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# straight circular fiber - built "from scratch" out of Pipe primitives
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def circular_fiber():
    rcore, rclad = 10, 30
    ncore, nclad = 1.445, 1.44
    res = 30
    core = waveguide.Pipe(ncore, "core", res, rcore)
    clad = waveguide.Pipe(nclad, "clad", 3 * res, rclad)
    fiber = waveguide.Waveguide([clad, core])
    fiber.z_invariant = True
    return fiber, ncore, nclad


def test_circular_fiber_mesh(circular_fiber):
    fiber, *_ = circular_fiber
    mesh = fiber.make_mesh()
    assert set(mesh.cell_sets) == {"core", "clad"}
    fiber.plot_mesh(mesh=mesh, verbose=False)


def test_circular_fiber_modes(circular_fiber, save_dir):
    fiber, ncore, nclad = circular_fiber
    wavelength = 1.55
    Nmax = 10
    prop = Propagator(wavelength, fiber, Nmax, save_dir=save_dir)
    effective_indices, modes = prop.solve_at(0)

    assert len(effective_indices) == Nmax
    assert modes.shape[0] == Nmax
    # effective indices come out sorted high -> low
    assert np.all(np.diff(effective_indices) <= 1e-9)
    # the fundamental is guided: nclad < neff < ncore
    assert nclad < effective_indices[0] < ncore
    # modes 1 & 2 are the near-degenerate LP11 pair
    assert effective_indices[1] == pytest.approx(effective_indices[2], abs=1e-4)
    assert effective_indices[0] > effective_indices[1]

    prop.plot_cfield(modes[1].astype(complex), res=1.0)


# --------------------------------------------------------------------------- #
# tapered square fiber - a BoxPipe that triples in width over its length
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tapered_box_fiber():
    ncore, nclad = 1.445, 1.44
    length = 10000
    xw_core = lambda z: 10 * (1 + 2 * z / length)
    yw_core = 10
    rect_core = waveguide.BoxPipe(ncore, "core", xw_core, yw_core)
    xw_clad = lambda z: 30 * (1 + 2 * z / length)
    rect_clad = waveguide.BoxPipe(nclad, "clad", xw_clad, 3 * yw_core)
    rect_clad.mesh_size = 3.0
    rect_core.mesh_size = 1.0
    rect_fiber = waveguide.Waveguide([rect_clad, rect_core])
    return rect_fiber, length


def test_tapered_box_fiber_mesh_widens_with_z(tapered_box_fiber):
    rect_fiber, length = tapered_box_fiber
    mesh0 = rect_fiber.make_mesh()
    meshL = rect_fiber.transform_mesh(mesh0, 0, length)
    w0 = np.ptp(mesh0.points[:, 0])
    wL = np.ptp(meshL.points[:, 0])
    assert wL == pytest.approx(3.0 * w0, rel=0.05)
    assert np.ptp(meshL.points[:, 1]) == pytest.approx(np.ptp(mesh0.points[:, 1]), rel=1e-6)


@pytest.mark.slow
def test_tapered_box_fiber_compute_neffs(tapered_box_fiber, save_dir):
    rect_fiber, length = tapered_box_fiber
    wavelength = 1.55
    rect_prop = Propagator(wavelength, rect_fiber, 6, save_dir=save_dir)
    zs, neffs = rect_prop.compute_neffs(0, length, save=False, tag="tapered_box")

    assert zs[0] == 0.0 and zs[-1] == length
    assert neffs.shape == (len(zs), 6)
    # fundamental mode stays the highest-index mode and becomes more bound as
    # the guide widens
    assert np.all(neffs[:, 0] >= neffs[:, 1] - 1e-9)
    assert neffs[-1, 0] > neffs[0, 0]

    rect_prop.plot_neffs()
    rect_prop.plot_waveguide_mode(2)
