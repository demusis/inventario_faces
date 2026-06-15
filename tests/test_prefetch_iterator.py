from __future__ import annotations

import unittest

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.services.pipeline_support import prefetch_iterator


class PrefetchIteratorTests(unittest.TestCase):
    def test_preserves_order_and_values(self) -> None:
        source = list(range(100))
        result = list(prefetch_iterator(iter(source), buffer_size=8))
        self.assertEqual(result, source)

    def test_small_buffer_still_yields_everything(self) -> None:
        source = list(range(50))
        result = list(prefetch_iterator(iter(source), buffer_size=1))
        self.assertEqual(result, source)

    def test_empty_source(self) -> None:
        self.assertEqual(list(prefetch_iterator(iter([]), buffer_size=4)), [])

    def test_producer_exception_is_propagated(self) -> None:
        def _explode():
            yield 1
            yield 2
            raise RuntimeError("falha de decodificacao")

        collected = []
        with self.assertRaises(RuntimeError):
            for value in prefetch_iterator(_explode(), buffer_size=2):
                collected.append(value)
        self.assertEqual(collected, [1, 2])

    def test_invalid_buffer_size(self) -> None:
        with self.assertRaises(ValueError):
            list(prefetch_iterator(iter([1]), buffer_size=0))

    def test_consumer_early_break_does_not_hang(self) -> None:
        produced = []

        def _counting():
            for value in range(1000):
                produced.append(value)
                yield value

        iterator = prefetch_iterator(_counting(), buffer_size=4)
        first = next(iterator)
        iterator.close()
        self.assertEqual(first, 0)


if __name__ == "__main__":
    unittest.main()
