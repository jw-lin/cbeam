"""Unit tests for ``cbeam.FEval`` (the Julia-backed field evaluator).

A quadratic (order-2) finite element mesh represents any *linear* field
``f(x, y) = a*x + b*y + c`` exactly, so these tests use a linear field and
check for exact recovery / constant gradient.
"""

import numpy as np
import pytest

from cbeam import waveguide as wg
from cbeam import FEval


@pytest.fixture(scope="module")
def mesh():
    f = wg.CircularStepIndexFiber(10, 30, 1.445, 1.44, core_res=16, clad_res=32)
    return f.make_mesh()


@pytest.fixture(scope="module")
def tree(mesh):
    return FEval.create_tree_from_mesh(mesh)


@pytest.fixture
def linear_field(mesh):
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    return 0.3 * x - 0.7 * y + 1.5


class TestTreeConstruction:
    def test_create_tree_from_mesh(self, mesh):
        t = FEval.create_tree_from_mesh(mesh)
        assert t is not None

    def test_create_tree_from_arrays(self, mesh):
        t = FEval.create_tree(mesh.points, mesh.cells[1].data)
        assert t is not None

    def test_sort_mesh_attaches_tree(self, mesh):
        import copy

        m = copy.deepcopy(mesh)
        assert not hasattr(m, "tree")
        FEval.sort_mesh(m)
        assert hasattr(m, "tree")


class TestQuery:
    def test_query_returns_valid_triangle_index(self, mesh, tree):
        ntri = mesh.cells[1].data.shape[0]
        idx = FEval.query(np.array([0.0, 0.0]), tree)
        assert 0 <= idx < ntri

    def test_query_origin_triangle_contains_origin(self, mesh, tree):
        idx = FEval.query(np.array([0.0, 0.0]), tree)
        tri = mesh.cells[1].data[idx]
        verts = mesh.points[tri[:3]]
        # barycentric check that the origin is inside the returned triangle
        (x1, y1), (x2, y2), (x3, y3) = verts
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        l1 = ((y2 - y3) * (0 - x3) + (x3 - x2) * (0 - y3)) / det
        l2 = ((y3 - y1) * (0 - x3) + (x1 - x3) * (0 - y3)) / det
        l3 = 1 - l1 - l2
        assert min(l1, l2, l3) >= -1e-9


class TestEvaluate:
    def test_recovers_linear_field_at_nodes(self, mesh, tree, linear_field):
        pts = mesh.points
        vals = FEval.evaluate(pts, linear_field, tree)
        assert np.allclose(vals, linear_field, atol=1e-7)

    def test_single_point(self, tree, linear_field, mesh):
        val = FEval.evaluate(np.array([1.0, 2.0]), linear_field, tree)
        assert float(val) == pytest.approx(0.3 * 1.0 - 0.7 * 2.0 + 1.5, abs=1e-7)

    def test_evaluate_grid_shape_and_values(self, tree, linear_field):
        xa = np.linspace(-5, 5, 7)
        ya = np.linspace(-4, 4, 5)
        grid = np.array(FEval.evaluate_grid(xa, ya, linear_field, tree))
        assert grid.shape == (7, 5)
        expected = 0.3 * xa[:, None] - 0.7 * ya[None, :] + 1.5
        assert np.allclose(grid, expected, atol=1e-6)


class TestResample:
    def test_resample_onto_same_mesh_is_identity(self, mesh, linear_field):
        out = FEval.resample(linear_field, mesh, mesh)
        assert np.allclose(out, linear_field, atol=1e-7)

    def test_resample_onto_coarser_mesh(self, mesh, linear_field):
        coarse = wg.CircularStepIndexFiber(
            10, 25, 1.445, 1.44, core_res=10, clad_res=16
        ).make_mesh()
        out = FEval.resample(linear_field, mesh, coarse)
        expected = 0.3 * coarse.points[:, 0] - 0.7 * coarse.points[:, 1] + 1.5
        assert np.allclose(out, expected, atol=1e-6)


class TestTransverseGradient:
    def test_constant_gradient_of_linear_field(self, mesh, linear_field):
        g = FEval.transverse_gradient(
            linear_field, mesh.cells[1].data, mesh.points
        )
        assert g.shape == (mesh.points.shape[0], 2)
        assert np.allclose(g[:, 0], 0.3, atol=1e-6)
        assert np.allclose(g[:, 1], -0.7, atol=1e-6)


class TestMeshAccessors:
    def test_get_triangles(self, mesh):
        tris = FEval.get_triangles(mesh)
        assert tris.shape[1] == 6
        assert np.array_equal(tris, mesh.cells[1].data)

    def test_get_points(self, mesh):
        pts = FEval.get_points(mesh)
        assert np.array_equal(pts, mesh.points)
