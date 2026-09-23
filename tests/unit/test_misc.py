"""Unit tests for the free helper functions in ``cbeam.waveguide``."""

import numpy as np
import pytest

from cbeam import waveguide as wg

# waveguide geometry / gmsh meshing is pinned to numpy regardless of
# CBEAM_BACKEND, so this module has no distinct jax code path.
pytestmark = pytest.mark.numpy_only


class TestGet19PortPositions:
    def test_shape_and_centre(self):
        pos = wg.get_19port_positions(2.0)
        assert pos.shape == (19, 2)
        # first core sits at the origin
        assert np.allclose(pos[0], [0.0, 0.0])

    def test_first_ring_radius(self):
        spacing = 3.5
        pos = wg.get_19port_positions(spacing)
        # cores 1..6 form a hexagon of radius == spacing
        ring = pos[1:7]
        radii = np.hypot(ring[:, 0], ring[:, 1])
        assert np.allclose(radii, spacing)

    def test_scales_linearly_with_spacing(self):
        a = wg.get_19port_positions(1.0)
        b = wg.get_19port_positions(2.0)
        assert np.allclose(2.0 * a, b)


class TestBlend:
    def test_limits(self):
        z = np.array([-1e6, 1e6])
        out = wg.blend(z, 0.0, 1.0)
        assert out[0] == pytest.approx(0.0, abs=1e-9)
        assert out[1] == pytest.approx(1.0, abs=1e-9)

    def test_midpoint_is_half(self):
        assert wg.blend(5.0, 5.0, 1.0) == pytest.approx(0.5)

    def test_monotonic_non_decreasing(self):
        z = np.linspace(-10, 10, 200)
        out = wg.blend(z, 0.0, 2.0)
        # tanh saturates far from the centre, so the curve is flat there but
        # never decreasing, and strictly rising through the transition
        assert np.all(np.diff(out) >= -1e-12)
        centre = (np.abs(z) < 1.0)
        assert np.all(np.diff(out[centre]) > 0)


class TestDist:
    def test_scalar(self):
        assert wg.dist(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_axis_reduction(self):
        p1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        p2 = np.array([[3.0, 4.0], [1.0, 1.0]])
        out = wg.dist(p1, p2, axis=1)
        assert np.allclose(out, [5.0, 0.0])


class TestRotate:
    def test_vector_90_degrees(self):
        out = wg.rotate(np.array([1.0, 0.0]), np.pi / 2)
        assert np.allclose(out, [0.0, 1.0], atol=1e-12)

    def test_matrix_input(self):
        v = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = wg.rotate(v, np.pi / 2)
        assert out.shape == (2, 2)
        assert np.allclose(out[0], [0.0, 1.0], atol=1e-12)
        assert np.allclose(out[1], [-1.0, 0.0], atol=1e-12)

    def test_rotation_preserves_norm(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=2)
        out = wg.rotate(v, 0.731)
        assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(v))


class TestLinearTaper:
    def test_endpoints(self):
        f = wg.linear_taper(3.0, 10.0)
        assert f(0.0) == pytest.approx(1.0)
        assert f(10.0) == pytest.approx(3.0)

    def test_is_linear(self):
        f = wg.linear_taper(5.0, 20.0)
        assert f(10.0) == pytest.approx(0.5 * (f(0.0) + f(20.0)))
