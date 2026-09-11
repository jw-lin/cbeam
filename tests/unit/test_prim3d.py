"""Unit tests for the 3D primitive classes in ``cbeam.waveguide``."""

import numpy as np
import pytest

from cbeam import waveguide as wg


class TestPrim3DBase:
    def test_init_and_index_passthrough(self):
        c = wg.Circle(1.47)
        c.make_points(1.0, 8)
        p = wg.Prim3D(c, "core")
        assert p.label == "core"
        assert p.n == 1.47
        assert p.preserve_shape is True

    def test_mesh_size_setter_propagates_to_prim2d(self):
        c = wg.Circle(1.47)
        c.make_points(1.0, 8)
        p = wg.Prim3D(c, "core")
        p.mesh_size = 0.3
        assert p.prim2D.mesh_size == 0.3
        p.skip_refinement = True
        assert p.prim2D.skip_refinement is True

    def test_update_moves_prim2d_points(self):
        pipe = wg.Pipe(1.5, "c", 12, lambda z: 1.0 + 0.01 * z)
        pipe.update(0.0)
        r0 = np.hypot(*pipe.prim2D.points.T).mean()
        pipe.update(100.0)
        r1 = np.hypot(*pipe.prim2D.points.T).mean()
        assert r1 == pytest.approx(2.0, rel=1e-6)
        assert r1 > r0


class TestPipe:
    def test_constant_radius_via_scalar(self):
        pipe = wg.Pipe(1.5, "core", 16, 2.5, (1.0, 0.0))
        pipe.update(0.0)
        pts = pipe.prim2D.points
        assert pts.shape == (16, 2)
        assert np.allclose(np.hypot(pts[:, 0] - 1.0, pts[:, 1]), 2.5)

    def test_callable_radius_and_center(self):
        pipe = wg.Pipe(1.5, "core", 20, lambda z: 1.0 + z, lambda z: (z, 2 * z))
        pipe.update(3.0)
        pts = pipe.prim2D.points
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        assert cx == pytest.approx(3.0, abs=1e-6)
        assert cy == pytest.approx(6.0, abs=1e-6)
        assert np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).mean() == pytest.approx(4.0, rel=1e-6)

    def test_transform_point_inside_scales_about_centre(self):
        pipe = wg.Pipe(1.5, "core", 16, lambda z: 1.0 + z, (0.0, 0.0))
        x1, y1 = pipe.transform_point_inside(
            np.array([0.5]), np.array([0.0]), 0.0, 1.0
        )
        # radius doubled (1 -> 2), so a point at r=0.5 moves to r=1.0
        assert x1[0] == pytest.approx(1.0)
        assert y1[0] == pytest.approx(0.0, abs=1e-12)


class TestLinearPipe:
    def test_radius_interpolates_linearly(self):
        lp = wg.LinearPipe(1.5, "c", 12, r1=1.0, r2=3.0, z_ex=10.0)
        assert lp.rfunc(0.0) == pytest.approx(1.0)
        assert lp.rfunc(10.0) == pytest.approx(3.0)
        assert lp.rfunc(5.0) == pytest.approx(2.0)

    def test_center_interpolates_linearly(self):
        lp = wg.LinearPipe(1.5, "c", 12, 1.0, 1.0, 10.0, c1=(0.0, 0.0), c2=(10.0, -20.0))
        assert lp.cfunc(5.0) == pytest.approx((5.0, -10.0))


class TestBox:
    def test_fixed_rectangular_cross_section(self):
        box = wg.Box(1.44, "clad", -5, 5, -3, 3)
        box.update(0.0)
        assert np.array_equal(
            box.prim2D.points, [[-5, -3], [5, -3], [5, 3], [-5, 3]]
        )
        # a Box does not change shape with z
        box.update(1234.0)
        assert np.array_equal(
            box.prim2D.points, [[-5, -3], [5, -3], [5, 3], [-5, 3]]
        )


class TestBoxPipe:
    def test_scalar_widths_centered(self):
        bp = wg.BoxPipe(1.5, "core", 4.0, 2.0, (0.0, 0.0))
        bp.update(0.0)
        assert np.array_equal(
            bp.prim2D.points, [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]]
        )

    def test_callable_widths_scale_with_z(self):
        bp = wg.BoxPipe(1.5, "core", lambda z: 2.0 + z, lambda z: 2.0, (0.0, 0.0))
        bp.update(2.0)
        pts = bp.prim2D.points
        width = pts[:, 0].max() - pts[:, 0].min()
        assert width == pytest.approx(4.0)

    def test_transform_point_inside_scales_each_axis(self):
        bp = wg.BoxPipe(1.5, "core", lambda z: 2.0 + 2.0 * z, lambda z: 2.0, (0.0, 0.0))
        x1, y1 = bp.transform_point_inside(
            np.array([1.0]), np.array([0.5]), 0.0, 1.0
        )
        # x width goes 2 -> 4 (scale 2), y width unchanged
        assert x1[0] == pytest.approx(2.0)
        assert y1[0] == pytest.approx(0.5)
