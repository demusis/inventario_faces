from __future__ import annotations

from threading import Event, Thread
import unittest

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.app import clear_shared_face_analyzer_cache, get_shared_face_analyzer
from inventario_faces.domain.config import FaceModelSettings


class FaceAnalyzerCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_shared_face_analyzer_cache()

    def tearDown(self) -> None:
        clear_shared_face_analyzer_cache()

    def test_shared_face_analyzer_reuses_same_instance_for_same_settings(self) -> None:
        settings = FaceModelSettings(
            backend="insightface",
            model_name="buffalo_l",
            det_size=(640, 640),
        )
        calls: list[str] = []

        def builder(model_settings: FaceModelSettings) -> object:
            calls.append(model_settings.model_name)
            return object()

        first = get_shared_face_analyzer(settings, builder=builder)
        second = get_shared_face_analyzer(settings, builder=builder)

        self.assertIs(first, second)
        self.assertEqual(["buffalo_l"], calls)

    def test_shared_face_analyzer_coalesces_concurrent_initialization(self) -> None:
        settings = FaceModelSettings(
            backend="insightface",
            model_name="buffalo_l",
            det_size=(640, 640),
        )
        start_event = Event()
        release_event = Event()
        calls: list[str] = []
        results: list[object] = []

        def builder(model_settings: FaceModelSettings) -> object:
            calls.append(model_settings.model_name)
            start_event.set()
            release_event.wait(timeout=5)
            return object()

        def target() -> None:
            results.append(get_shared_face_analyzer(settings, builder=builder))

        first_thread = Thread(target=target)
        second_thread = Thread(target=target)
        first_thread.start()
        start_event.wait(timeout=5)
        second_thread.start()
        release_event.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)

        self.assertEqual(2, len(results))
        self.assertIs(results[0], results[1])
        self.assertEqual(["buffalo_l"], calls)


if __name__ == "__main__":
    unittest.main()
