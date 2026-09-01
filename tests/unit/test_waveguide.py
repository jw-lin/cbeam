"""Unit tests for the base ``cbeam.waveguide.Waveguide`` class."""

import numpy as np
import pytest

from cbeam import waveguide as wg


@pytest.fixture
def simple_fiber():
    """A plain 2-layer circular fiber built from raw ``Pipe`` primitives
    (this is the "from scratch" fiber from ``docs_source/fib.rst``)."""
    core = wg.Pipe(1.445, "core", 24, 10.0)
    clad = wg.Pipe(1.44, "clad", 48, 30.0)
    fiber = wg.Waveguide([clad, core])
    fiber.z_invariant = True
    return fiber


class TestConstruction:
    def test_flattens_primitives(self, simple_fiber):
        assert len(simple_fiber.primsflat) == 2
        assert len(simple_fiber.prim3Dsflat) == 2
        assert {p.label for p in simple_fiber.prim3Dsflat} == {"core", "clad"}

    def test_nested_groups_are_flattened(self):
        clad = wg.Pipe(1.44, "clad", 32, 20.0)
        c1 = wg.Pipe(1.45, "core0", 16, 2.0, (-5.0, 0.0))
        c2 = wg.Pipe(1.45, "core1", 16, 2.0, (5.0, 0.0))
        w = wg.Waveguide([clad, [c1, c2]])
        assert len(w.prim3Dsflat) == 3
        assert len(w.primsflat) == 3

    def test_update_at_init_sets_z0_points(self, simple_fiber):
        # core primitive points should describe a radius-10 circle at z = 0
        core = simple_fiber.prim3Dsflat[1]
        r = np.hypot(*core.prim2D.points.T)
        assert np.allclose(r, 10.0)


class TestIORDict:
    def test_assign_ior_maps_labels(self, simple_fiber):
        d = simple_fiber.assign_IOR()
        assert d == {"clad": 1.44, "core": 1.445}

    def test_isolate_sets_other_channels_to_background(self):
        clad = wg.Pipe(1.44, "clad", 32, 20.0)
        c0 = wg.Pipe(1.45, "core0", 16, 2.0, (-5.0, 0.0))
        c1 = wg.Pipe(1.45, "core1", 16, 2.0, (5.0, 0.0))
        w = wg.Waveguide([clad, [c0, c1]])
        d = w.isolate(0)
        assert d["core0"] == 1.45          # kept
        assert d["core1"] == pytest.approx(1.44)   # pushed to background


class TestMeshGeneration:
    def test_make_mesh_returns_2d_points(self, simple_fiber):
        mesh = simple_fiber.make_mesh()
        assert mesh.points.shape[1] == 2
        assert mesh.points.shape[0] > 50
        assert set(mesh.cell_sets) == {"core", "clad"}

    def test_make_mesh_writes_file(self, simple_fiber, tmp_path):
        target = tmp_path / "mymesh"
        simple_fiber.make_mesh_bndry_ref(writeto=str(target))
        assert target.with_suffix(".msh").exists()

    def test_compute_mesh_size_within_bounds(self, simple_fiber):
        simple_fiber.min_mesh_size = 0.05
        simple_fiber.max_mesh_size = 5.0
        s = simple_fiber._compute_mesh_size(
            0.0, 0.0, min_size=simple_fiber.min_mesh_size,
            max_size=simple_fiber.max_mesh_size,
        )
        assert s >= simple_fiber.min_mesh_size


class TestTransform:
    def test_z_invariant_transform_is_identity(self, simple_fiber):
        x0 = np.array([1.0, 2.0, 3.0])
        y0 = np.array([0.0, -1.0, 2.0])
        x, y = simple_fiber.transform(x0, y0, 0.0, 5000.0)
        assert np.allclose(x, x0)
        assert np.allclose(y, y0)

    def test_transform_mesh_identity_when_z_unchanged(self, simple_fiber):
        mesh = simple_fiber.make_mesh()
        out = simple_fiber.transform_mesh(mesh, 0.0, 0.0)
        assert out is mesh

    def test_transform_mesh_identity_when_z_invariant(self, simple_fiber):
        mesh = simple_fiber.make_mesh()
        out = simple_fiber.transform_mesh(mesh, 0.0, 1234.0)
        assert np.allclose(out.points, mesh.points)

    def test_tapered_transform_scales_mesh(self):
        # a linearly widening box fiber -> mesh at z=L is wider than at z=0
        length = 10000.0
        core = wg.BoxPipe(1.445, "core", lambda z: 10 * (1 + 2 * z / length), 10.0)
        clad = wg.BoxPipe(1.44, "clad", lambda z: 30 * (1 + 2 * z / length), 30.0)
        core.mesh_size, clad.mesh_size = 1.0, 3.0
        fiber = wg.Waveguide([clad, core])

        mesh0 = fiber.make_mesh()
        width0 = mesh0.points[:, 0].max() - mesh0.points[:, 0].min()
        meshL = fiber.transform_mesh(mesh0, 0.0, length)
        widthL = meshL.points[:, 0].max() - meshL.points[:, 0].min()

        assert widthL == pytest.approx(3.0 * width0, rel=0.05)
        # y extent must be untouched
        assert np.ptp(meshL.points[:, 1]) == pytest.approx(
            np.ptp(mesh0.points[:, 1]), rel=1e-6
        )

    def test_deriv_transform_zero_for_z_invariant(self, simple_fiber):
        x0 = np.array([1.0, 2.0])
        y0 = np.array([0.0, 1.0])
        dx, dy = simple_fiber.deriv_transform(x0, y0, 0.0, 100.0)
        assert np.allclose(dx, 0.0)
        assert np.allclose(dy, 0.0)


class TestPlotting:
    def test_plot_mesh_runs(self, simple_fiber):
        mesh = simple_fiber.make_mesh()
        simple_fiber.plot_mesh(mesh=mesh, verbose=False)

    def test_module_level_plot_mesh_runs(self, simple_fiber):
        mesh = simple_fiber.make_mesh()
        wg.plot_mesh(mesh, IOR_dict=simple_fiber.assign_IOR(), verbose=False)

    def test_plot_boundaries_runs(self, simple_fiber):
        simple_fiber.plot_boundaries()
