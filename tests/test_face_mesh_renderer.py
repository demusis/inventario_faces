from __future__ import annotations

import unittest

import numpy as np

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.infrastructure.face_mesh_renderer import (
    build_face_mesh_geometry,
    draw_face_mesh,
)


class FaceMeshRendererTests(unittest.TestCase):
    def test_build_face_mesh_geometry_applies_translation_and_returns_unique_edges(self) -> None:
        points, edges = build_face_mesh_geometry(
            ((10.0, 10.0), (30.0, 10.0), (20.0, 25.0), (20.0, 25.0)),
            width=80,
            height=80,
            translate=(5.0, -2.0),
        )

        self.assertEqual([(15, 8), (35, 8), (25, 23)], points)
        self.assertTrue(edges)
        self.assertEqual(len(edges), len(set(edges)))

    def test_draw_face_mesh_uses_fixed_point_footprint_across_canvas_sizes(self) -> None:
        small = np.zeros((80, 80, 3), dtype=np.uint8)
        large = np.zeros((400, 400, 3), dtype=np.uint8)

        small_mesh = draw_face_mesh(small, ((20.0, 20.0),), draw_bbox=False)
        large_mesh = draw_face_mesh(large, ((20.0, 20.0),), draw_bbox=False)

        small_changed = int(np.count_nonzero(np.any(small_mesh != 0, axis=2)))
        large_changed = int(np.count_nonzero(np.any(large_mesh != 0, axis=2)))

        self.assertEqual(small.shape, small_mesh.shape)
        self.assertEqual(large.shape, large_mesh.shape)
        self.assertEqual(small_changed, large_changed)
        self.assertGreater(small_changed, 0)


if __name__ == "__main__":
    unittest.main()
