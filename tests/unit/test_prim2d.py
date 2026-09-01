"""Unit tests for the 2D primitive classes in ``cbeam.waveguide``."""

import numpy as np
import pytest

from cbeam import waveguide as wg


class TestPrim2DBase:
    def test_init_stores_index_and_points(self):
        pts = [(0, 0), (1, 0), (1, 1)]
        p = wg.Prim2D(1.45, pts)
        assert p.n == 1.45
        assert p.res == 3
        assert p.mesh_size is None
        assert p.skip_refinement is False

    def test_update_replaces_points(self):
        p = wg.Prim2D(1.0, [(0, 0), (1, 1)])
        newp = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        out = p.update(newp)
        assert np.array_equal(out, newp)
        assert p.res == 4

    def test_make_points_returns_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        p = wg.Prim2D(1.0, pts)
        assert np.array_equal(p.make_points(), pts)


class TestCircle:
    def test_make_points_count_and_radius(self):
        c = wg.Circle(1.5)
        pts = c.make_points(2.0, 12, center=(1.0, -1.0))
        assert pts.shape == (12, 2)
        radii = np.hypot(pts[:, 0] - 1.0, pts[:, 1] + 1.0)
        assert np.allclose(radii, 2.0)
        assert c.radius == 2.0
        assert c.center == (1.0, -1.0)

    def test_boundary_dist_sign(self):
        c = wg.Circle(1.5)
        c.make_points(2.0, 16)
        assert c.boundary_dist(0.0, 0.0) == pytest.approx(-2.0)   # centre -> inside
        assert c.boundary_dist(2.0, 0.0) == pytest.approx(0.0)    # on the edge
        assert c.boundary_dist(5.0, 0.0) == pytest.approx(3.0)    # outside

    def test_boundary_dist_vectorized(self):
        c = wg.Circle(1.5)
        c.make_points(1.0, 16)
        x = np.array([0.0, 1.0, 3.0])
        y = np.zeros_like(x)
        assert np.allclose(c.boundary_dist(x, y), [-1.0, 0.0, 2.0])

    def test_nearest_boundary_point_on_circle(self):
        c = wg.Circle(1.5)
        c.make_points(2.0, 16, center=(0.0, 0.0))
        bx, by = c.nearest_boundary_point(10.0, 0.0)
        assert (bx, by) == pytest.approx((2.0, 0.0))
        bx, by = c.nearest_boundary_point(0.0, -10.0)
        assert (bx, by) == pytest.approx((0.0, -2.0))


class TestRectangle:
    def test_make_points_corners(self):
        r = wg.Rectangle(2.0)
        pts = r.make_points(-1, 1, -2, 2)
        assert np.array_equal(
            pts, [[-1, -2], [1, -2], [1, 2], [-1, 2]]
        )
        assert r.bounds == [-1, 1, -2, 2]

    def test_boundary_dist_inside_is_negative(self):
        r = wg.Rectangle(2.0)
        r.make_points(-2, 2, -2, 2)
        assert r.boundary_dist(0.0, 0.0) == pytest.approx(-2.0)
        # closest edge, not the corner
        assert r.boundary_dist(1.5, 0.0) == pytest.approx(-0.5)

    def test_boundary_dist_outside_is_positive(self):
        r = wg.Rectangle(2.0)
        r.make_points(-1, 1, -1, 1)
        d = r.boundary_dist(np.array([3.0]), np.array([0.0]))
        assert d > 0

    def test_nearest_boundary_point_edge(self):
        r = wg.Rectangle(2.0)
        r.make_points(-1, 1, -1, 1)
        bx, by = r.nearest_boundary_point(np.array([0.0]), np.array([5.0]))
        assert by == pytest.approx(1.0)
        assert -1.0 <= bx <= 1.0

    @pytest.mark.xfail(
        reason="known bug: Rectangle.nearest_boundary_point writes outx instead "
        "of outy in the 'i==2, x>xmax' branch (waveguide.py ~L321), so the "
        "returned point is not actually on the rectangle boundary",
        strict=False,
    )
    def test_nearest_boundary_point_beyond_right_edge(self):
        r = wg.Rectangle(2.0)
        r.make_points(-1, 1, -2, 2)
        bx, by = r.nearest_boundary_point(np.array([5.0]), np.array([0.0]))
        # closest boundary point to (5, 0) is the right edge at (1, 0)
        assert (float(bx), float(by)) == pytest.approx((1.0, 0.0))


class TestPrim2DUnion:
    def test_requires_matching_index(self):
        c1 = wg.Circle(1.5)
        c2 = wg.Circle(1.6)
        with pytest.raises(AssertionError):
            wg.Prim2DUnion(c1, c2)

    def test_make_points_stacks_both(self):
        c1, c2 = wg.Circle(1.5), wg.Circle(1.5)
        u = wg.Prim2DUnion(c1, c2)
        pts = u.make_points((1.0, 8), (1.0, 8, (5.0, 0.0)))
        assert pts.shape == (2, 8, 2)

    def test_boundary_dist_is_min_of_parts(self):
        c1, c2 = wg.Circle(1.5), wg.Circle(1.5)
        c1.make_points(1.0, 16, center=(0.0, 0.0))
        c2.make_points(1.0, 16, center=(5.0, 0.0))
        u = wg.Prim2DUnion(c1, c2)
        # a point just outside c1 but far from c2
        assert u.boundary_dist(2.0, 0.0) == pytest.approx(
            min(c1.boundary_dist(2.0, 0.0), c2.boundary_dist(2.0, 0.0))
        )
