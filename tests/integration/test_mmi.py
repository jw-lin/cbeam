"""Integration test rewritten from ``docs_source/mmi.rst``.

Multimode-interference coupler: solve the access waveguide, characterize the
(z-invariant) slab, move the launch field between the two meshes with
``FEval.resample``, and check the self-imaging distance produces a 3-fold image.
Everything here is cheap (both waveguides are z-invariant).
"""

import numpy as np
import pytest
from scipy.signal import find_peaks

from cbeam.waveguide import RectangularStepIndexFiber
from cbeam.propagator import Propagator
from cbeam import FEval

pytestmark = pytest.mark.integration

WL = 1.55
NCORE, NCLAD = 1.445, 1.44


@pytest.fixture(scope="module")
def guides():
    mmi_width, mmi_height = 60.0, 6.0
    access_width = access_height = 6.0
    access = RectangularStepIndexFiber(
        access_width, access_height, access_height * 6, access_width * 6,
        NCORE, NCLAD, 1.0, 5.0,
    )
    mmi = RectangularStepIndexFiber(
        mmi_width, mmi_height, mmi_width * 3, mmi_height * 6,
        NCORE, NCLAD, 1.0, 5.0,
    )
    return access, mmi


def test_plot_meshes(guides):
    access, mmi = guides
    access.plot_mesh(verbose=False)
    mmi.plot_mesh(verbose=False)


def test_access_mode_solve(guides, save_dir, golden):
    access, _ = guides
    access_prop = Propagator(WL, access, Nmax=1, save_dir=save_dir)
    ac_neffs, ac_modes = access_prop.solve_at(0)
    assert ac_modes.shape[0] == 1
    assert NCLAD < ac_neffs[0] < NCORE
    golden.check("neff", ac_neffs, rtol=1e-6, atol=1e-9)
    access_prop.plot_cfield(ac_modes[0].astype(complex), show_mesh=True)


def test_resample_and_self_image(guides, save_dir, golden):
    access, mmi = guides

    access_prop = Propagator(WL, access, Nmax=1, save_dir=save_dir + "/acc")
    ac_neffs, ac_modes = access_prop.solve_at(0)

    mmi_prop = Propagator(WL, mmi, Nmax=8, save_dir=save_dir + "/mmi")
    mmi_prop.characterize(save=False)
    assert mmi_prop.neffs.shape == (1, 8)
    golden.check("mmi_neffs", mmi_prop.neffs[0], rtol=1e-6, atol=1e-9)
    # z-invariant guide -> no coupling
    assert mmi_prop.cmats is None or np.allclose(mmi_prop.cmats, 0.0)

    # move the launch field from the access mesh onto the slab mesh
    launch_field = FEval.resample(ac_modes[0], access_prop.mesh, mmi_prop.mesh)
    assert launch_field.shape[0] == mmi_prop.mesh.points.shape[0]
    assert np.isfinite(launch_field).all()
    mmi_prop.plot_cfield(launch_field.astype(complex),
                         xlim=(-16, 16), ylim=(-16, 16), show_mesh=True)

    launch_modes = mmi_prop.make_mode_vector(launch_field)
    assert launch_modes.shape == (8,)
    # doc testoutput (up to an overall eigenmode sign, which is mesh dependent):
    #   [6.6e-1, ~0, -5.3e-1, ~0, -3.8e-1, ~0, 2.6e-1, ~0]
    # even modes dominate, odd modes ~ 0, total power < 1 (radiative loss)
    assert abs(launch_modes[0]) == pytest.approx(0.660, abs=0.05)
    assert abs(launch_modes[2]) == pytest.approx(0.533, abs=0.05)
    assert abs(launch_modes[4]) == pytest.approx(0.380, abs=0.05)
    assert abs(launch_modes[1]) < 1e-2
    assert abs(launch_modes[3]) < 1e-2
    total_power = np.sum(np.abs(launch_modes) ** 2)
    assert total_power < 1.0
    assert total_power == pytest.approx(0.93, abs=0.05)
    golden.check("launch_modes_abs", launch_modes, abs_compare=True, atol=1e-3)

    # three-fold self image at L = 3 * Lpi / (4 * N),  N = 3
    betas = mmi_prop.neffs[0] * 2 * np.pi / WL
    L = np.pi / (betas[0] - betas[1])
    f = mmi_prop.make_field(launch_modes, z=L / 4, apply_phase=True)
    assert np.isfinite(f).all()

    # the field at z = L/4 should have (roughly) three lobes along x: sample the
    # intensity on the core mid-line and count peaks
    xs = np.linspace(-25, 25, 401)
    tree = FEval.create_tree_from_mesh(mmi_prop.mesh)
    line = np.column_stack([xs, np.zeros_like(xs)])
    inten = np.abs(FEval.evaluate(line, np.real(f), tree)) ** 2
    inten += np.abs(FEval.evaluate(line, np.imag(f), tree)) ** 2
    if inten.max() > 0:
        inten /= inten.max()
    # count peaks above 25% of the max intensity; require a minimum separation
    # of 15 samples (~1.9 units) so a single broad lobe isn't double-counted -
    # the lobes are ~20 units apart, well above that
    peak_idx, _ = find_peaks(inten, height=0.25, distance=15)
    assert len(peak_idx) == 3
