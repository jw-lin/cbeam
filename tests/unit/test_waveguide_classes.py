"""Unit tests for the pre-defined ``Waveguide`` subclasses.

These build each waveguide with deliberately coarse meshing so the tests stay
fast; the physics is exercised in ``tests/integration``.
"""

import numpy as np
import pytest

from cbeam import waveguide as wg


class TestCircularStepIndexFiber:
    def test_is_z_invariant_step_index(self):
        f = wg.CircularStepIndexFiber(10, 30, 1.445, 1.44, core_res=12, clad_res=24)
        assert f.z_invariant is True
        assert f.assign_IOR() == {"clad": 1.44, "core": 1.445}

    def test_mesh_has_both_regions(self):
        f = wg.CircularStepIndexFiber(10, 30, 1.445, 1.44, core_res=12, clad_res=24)
        mesh = f.make_mesh()
        assert set(mesh.cell_sets) == {"core", "clad"}


class TestRectangularStepIndexFiber:
    def test_construction(self):
        f = wg.RectangularStepIndexFiber(
            6, 6, 36, 36, 1.445, 1.44, core_mesh_size=1.5, clad_mesh_size=6.0
        )
        assert f.z_invariant is True
        assert f.assign_IOR() == {"clad": 1.44, "core": 1.445}
        assert f.make_mesh().points.shape[1] == 2


@pytest.fixture(scope="module")
def small_lantern():
    core_pos = np.array(
        [[0, 0]]
        + [[6 * np.cos(i * 2 * np.pi / 5), 6 * np.sin(i * 2 * np.pi / 5)] for i in range(5)]
    )
    return wg.PhotonicLantern(
        core_pos, [0.3] * 6, 10, 30, [1.4528] * 6, 1.444, 1.4385,
        z_ex=40000, taper_factor=8.0,
        core_res=12, clad_res=24, jack_res=12,
        core_mesh_size=1.0, clad_mesh_size=3.0,
    )


class TestPhotonicLantern:
    def test_flags_and_length(self, small_lantern):
        assert small_lantern.linear is True
        assert small_lantern.z_ex == 40000

    def test_ior_dict(self, small_lantern):
        d = small_lantern.assign_IOR()
        assert d["jacket"] == pytest.approx(1.4385)
        assert d["cladding"] == pytest.approx(1.444)
        assert d["core0"] == pytest.approx(1.4528)
        assert sum(k.startswith("core") for k in d) == 6

    def test_transform_scales_about_origin(self, small_lantern):
        # PhotonicLantern overrides transform() with a pure taper scaling
        x0 = np.array([1.0, -2.0])
        y0 = np.array([0.5, 3.0])
        x, y = small_lantern.transform(x0, y0, 0.0, small_lantern.z_ex)
        scale = small_lantern.taper_func(small_lantern.z_ex) / small_lantern.taper_func(0.0)
        assert np.allclose(x, x0 * scale)
        assert np.allclose(y, y0 * scale)
        assert scale == pytest.approx(small_lantern.taper_factor)

    def test_deriv_transform_matches_finite_difference(self, small_lantern):
        x0 = np.array([1.0, -2.0])
        y0 = np.array([0.5, 3.0])
        z = 10000.0
        eps = 1.0
        xa, ya = small_lantern.transform(x0, y0, 0.0, z - eps)
        xb, yb = small_lantern.transform(x0, y0, 0.0, z + eps)
        dx_fd = 0.5 * (xb - xa) / eps
        dx, dy = small_lantern.deriv_transform(x0, y0, 0.0, z)
        assert np.allclose(dx, dx_fd, rtol=1e-6)

    def test_isolate_pushes_other_cores_to_cladding(self, small_lantern):
        d = small_lantern.isolate(2)
        assert d["core2"] == pytest.approx(1.4528)
        for k in range(6):
            if k != 2:
                assert d[f"core{k}"] == pytest.approx(1.444)

    def test_mesh_builds(self, small_lantern):
        mesh = small_lantern.make_mesh()
        assert mesh.points.shape[0] > 200
        assert "core0" in mesh.cell_sets


class TestTestPhotonicLantern:
    def test_matches_manual_six_port_lantern(self):
        lant = wg.TestPhotonicLantern()
        assert lant.z_ex == 40000
        assert lant.taper_factor == 8.0
        d = lant.assign_IOR()
        assert sum(k.startswith("core") for k in d) == 6


class TestDicoupler:
    @pytest.fixture(scope="class")
    def dc(self):
        return wg.Dicoupler(
            3.0, 3.0, 1.4528, 1.4528, dmax=60.0, dmin=10.0, nclad=1.444,
            coupling_length=5000.0, a=1250.0, core_res=12,
            core_mesh_size=1.5, clad_mesh_size=20.0,
        )

    def test_length_is_twice_coupling_length(self, dc):
        assert dc.z_ex == 10000.0

    def test_channel_separation_shrinks_then_grows(self, dc):
        d0 = dc.dfunc(0.0)
        dmid = dc.dfunc(dc.z_ex / 2)
        dend = dc.dfunc(dc.z_ex)
        assert d0 == pytest.approx(60.0, abs=1.0)
        assert dmid == pytest.approx(10.0, abs=1.0)
        assert dend == pytest.approx(60.0, abs=1.0)

    def test_channels_are_symmetric(self, dc):
        for z in (0.0, 2500.0, 5000.0):
            assert dc.c1func(z)[0] == pytest.approx(-dc.c2func(z)[0])

    def test_ior_and_mesh(self, dc):
        d = dc.assign_IOR()
        assert d["core1"] == pytest.approx(1.4528)
        assert d["cladding"] == pytest.approx(1.444)
        assert dc.make_mesh().points.shape[1] == 2

    def test_plot_paths_runs(self, dc):
        dc.plot_paths()


class TestTricoupler:
    @pytest.fixture(scope="class")
    def tri(self):
        return wg.Tricoupler(
            3.0, 1.4528, dmax=40.0, dmin=12.0, nclad=1.444,
            coupling_length=4000.0, a=1000.0, core_res=12,
            core_mesh_size=1.5, clad_mesh_size=15.0,
        )

    def test_three_cores_equilateral(self, tri):
        z = 2000.0
        c1 = np.array([tri.c1func(z)[0], 0.0])
        c2 = np.array(tri.c2func(z))
        c3 = np.array(tri.c3func(z))
        d12 = np.linalg.norm(c1 - c2)
        d23 = np.linalg.norm(c2 - c3)
        d31 = np.linalg.norm(c3 - c1)
        assert d12 == pytest.approx(d23, rel=1e-6)
        assert d23 == pytest.approx(d31, rel=1e-6)

    def test_ior_has_three_cores(self, tri):
        d = tri.assign_IOR()
        assert {"core1", "core2", "core3", "cladding"} <= set(d)

    def test_plot_paths_runs(self, tri):
        tri.plot_paths()


class TestPlanarTricoupler:
    def test_middle_core_on_axis(self):
        ptri = wg.PlanarTricoupler(
            3.0, 3.0, 1.4528, dmax=40.0, dmin=12.0, nclad=1.444,
            coupling_length=4000.0, a=1000.0, core_res=12,
            core_mesh_size=1.5, clad_mesh_size=15.0,
        )
        assert ptri.c3func(1234.0) == (0, 0)
        # outer channels are mirror images across x = 0
        z = 2000.0
        assert ptri.c2func(z)[0] == pytest.approx(-ptri.c1func(z)[0])
        assert ptri.z_ex == 8000.0


class TestOAMPhotonicLantern:
    def test_construction(self):
        oam = wg.OAMPhotonicLantern(
            ring_radius=10.0, ring_width=4.0, rcores=[0.4] * 5, rjack=30,
            ncores=[1.45] * 5, nclad=1.444, njack=1.4385, z_ex=40000,
            taper_factor=8.0, core_res=12, clad_res_outer=24, clad_res_inner=18,
            jack_res=12, core_mesh_size=1.0, clad_mesh_size=3.0,
            inner_clad_mesh_size=5.0,
        )
        assert oam.z_ex == 40000
        d = oam.assign_IOR()
        assert "inner_cladding" in d
        assert sum(k.startswith("core") for k in d) == 5
