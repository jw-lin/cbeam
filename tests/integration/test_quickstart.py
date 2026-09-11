"""Integration test rewritten from ``docs_source/basicusage.rst`` (quickstart).

Covers: build a pre-defined waveguide -> mesh + plot -> Propagator ->
characterize -> propagate -> make_field -> plot.
"""

import numpy as np
import pytest

from cbeam.propagator import Propagator
from cbeam.waveguide import TestPhotonicLantern as SixPortLantern

pytestmark = pytest.mark.integration


def test_waveguide_basics_make_and_plot_mesh():
    wvg = SixPortLantern()
    mesh = wvg.make_mesh()
    assert mesh.points.shape[1] == 2
    assert mesh.points.shape[0] > 1000
    # "can also leave out mesh; a mesh will be auto-generated"
    wvg.plot_mesh(mesh=mesh, verbose=False)


@pytest.mark.slow
def test_putting_it_all_together(save_dir, golden):
    wvg = SixPortLantern()
    wavelength = 1.55
    num_modes = 6

    prop = Propagator(wavelength, wvg, num_modes, save_dir=save_dir)
    prop.z_acc = -1.0

    # characterize a short leading section of the lantern (the doc runs the full
    # 40 mm device which takes ~80 s; 4 mm exercises the identical code path)
    zs, neffs, vs, cmats = prop.characterize(0, 4000, save=False)
    assert zs[0] == 0.0 and zs[-1] == 4000
    assert neffs.shape[1] == num_modes
    assert cmats.shape[1:] == (num_modes, num_modes)
    golden.check_vs_z("neffs", zs, neffs)

    u0 = [1, 0, 0, 0, 0, 0]
    zs, us, uf = prop.propagate(u0)
    assert us.shape[1] == num_modes
    assert np.sum(np.abs(uf) ** 2) == pytest.approx(1.0, abs=5e-3)
    golden.check("uf_power", np.abs(uf) ** 2, sort=True, atol=2e-3)

    output_field = prop.make_field(us[-1], zs[-1])
    assert output_field.shape[0] == prop.mesh.points.shape[0]
    assert np.iscomplexobj(output_field)

    prop.plot_cfield(output_field, z=zs[-1], show_mesh=True,
                     xlim=(-100, 100), ylim=(-100, 100), res=2.0)
    prop.plot_wavefront(zs, us)
