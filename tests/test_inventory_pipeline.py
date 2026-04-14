from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import (
    AppConfig,
    AppSettings,
    ClusteringSettings,
    FaceModelSettings,
    ForensicsSettings,
    LikelihoodRatioSettings,
    MediaSettings,
    ReportingSettings,
    VideoSettings,
)
from inventario_faces.domain.entities import BoundingBox, DetectedFace, ReportArtifacts, SampledFrame
from inventario_faces.services.clustering_service import ClusteringService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.inventory_service import InventoryService
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.video_service import VideoSamplingInfo, VideoService


class _FakeAnalyzer:
    providers = ["CPUExecutionProvider"]

    def analyze(self, frame: SampledFrame) -> list[DetectedFace]:
        crop = frame.bgr_pixels[5:40, 5:40].copy()
        return [
            DetectedFace(
                bbox=BoundingBox(5, 5, 40, 40),
                detection_score=0.97,
                embedding=[1.0, 0.0, 0.0],
                crop_bgr=crop,
                landmarks=((12.0, 16.0), (31.0, 16.0), (22.0, 24.0), (15.0, 32.0), (29.0, 32.0)),
                biometric_landmarks=(
                    (12.0, 16.0),
                    (20.0, 14.0),
                    (31.0, 16.0),
                    (17.0, 23.0),
                    (22.0, 24.0),
                    (28.0, 23.0),
                    (15.0, 32.0),
                    (22.0, 34.0),
                    (29.0, 32.0),
                ),
            )
        ]


class _CalibrationAwareAnalyzer(_FakeAnalyzer):
    _EMBEDDINGS = {
        "alice_1": [1.00, 0.02, 0.00],
        "alice_2": [0.99, 0.05, 0.00],
        "alice_3": [0.97, 0.08, 0.00],
        "alice_4": [0.96, 0.11, 0.00],
        "bob_1": [0.15, 0.99, 0.00],
        "bob_2": [0.18, 0.97, 0.00],
        "bob_3": [0.22, 0.95, 0.00],
        "bob_4": [0.25, 0.93, 0.00],
        "query_a": [0.98, 0.06, 0.00],
        "query_b": [0.97, 0.07, 0.00],
    }

    def analyze(self, frame: SampledFrame) -> list[DetectedFace]:
        detected = super().analyze(frame)
        embedding = self._EMBEDDINGS.get(Path(frame.source_path).stem.lower(), [1.0, 0.0, 0.0])
        detected[0].embedding = list(embedding)
        return detected


class _FakeReportGenerator:
    def generate(self, result) -> ReportArtifacts:
        report_dir = result.run_directory / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        tex_path = report_dir / "relatorio_forense.tex"
        tex_path.write_text("relatorio de teste", encoding="utf-8")
        return ReportArtifacts(tex_path=tex_path, pdf_path=None)


class _FakeFaceSearchReportGenerator:
    def generate(self, result) -> ReportArtifacts:
        report_dir = result.inventory_result.run_directory / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        tex_path = report_dir / "relatorio_busca_por_face.tex"
        docx_path = report_dir / "relatorio_busca_por_face.docx"
        tex_path.write_text("relatorio de busca por face", encoding="utf-8")
        docx_path.write_text("relatorio de busca por face", encoding="utf-8")
        return ReportArtifacts(tex_path=tex_path, pdf_path=None, docx_path=docx_path)


class _FakeMediaInfoExtractor:
    def extract(self, path: Path):
        from inventario_faces.domain.entities import MediaInfoAttribute, MediaInfoTrack

        return (
            (
                MediaInfoTrack(
                    track_type="Geral",
                    attributes=(
                        MediaInfoAttribute(label="Formato", value="JPEG"),
                        MediaInfoAttribute(label="Largura", value="80 pixels"),
                    ),
                ),
            ),
            None,
        )


class _FakeVideoService(VideoService):
    def __init__(self, settings: VideoSettings, frames: list[SampledFrame], sampling_info: VideoSamplingInfo) -> None:
        super().__init__(settings)
        self._frames = list(frames)
        self._sampling_info = sampling_info

    def sample_video(self, path: Path, metadata_callback=None):
        if metadata_callback is not None:
            metadata_callback(self._sampling_info)
        for frame in self._frames:
            yield frame


class _RecordingImageVideoService(VideoService):
    def __init__(self, settings: VideoSettings) -> None:
        super().__init__(settings)
        self.loaded_image_paths: list[Path] = []

    def load_image(self, path: Path) -> SampledFrame:
        self.loaded_image_paths.append(Path(path))
        return super().load_image(path)


class _FailOnceAnalyzer(_FakeAnalyzer):
    remaining_failures = 1

    def analyze(self, frame: SampledFrame) -> list[DetectedFace]:
        if _FailOnceAnalyzer.remaining_failures > 0:
            _FailOnceAnalyzer.remaining_failures -= 1
            raise RuntimeError("falha controlada")
        return super().analyze(frame)


class InventoryPipelineTests(unittest.TestCase):
    def test_pipeline_processes_image_and_exports_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            self._create_test_image(image_path)
            (root / "notes.txt").write_text("irrelevante", encoding="utf-8")

            service = InventoryService(
                config=self._config(),
                scanner_service=ScannerService(self._config().media),
                hashing_service=HashingService(),
                media_service=VideoService(self._config().video),
                clustering_service=ClusteringService(self._config().clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
                media_info_extractor=_FakeMediaInfoExtractor(),
            )

            result = service.run(root)

            self.assertEqual(2, result.summary.total_files)
            self.assertEqual(1, result.summary.image_files)
            self.assertEqual(1, result.summary.total_occurrences)
            self.assertEqual(1, result.summary.total_tracks)
            self.assertEqual(1, result.summary.total_keyframes)
            self.assertEqual(1, result.summary.total_clusters)
            self.assertEqual(1, len(result.files[0].media_info_tracks))
            self.assertIsNone(result.files[0].media_info_error)
            self.assertEqual(1, result.summary.total_detected_face_sizes.count)
            self.assertEqual(35.0, result.summary.total_detected_face_sizes.min_pixels)
            self.assertEqual(1, result.summary.selected_face_sizes.count)
            self.assertEqual(35.0, result.summary.selected_face_sizes.mean_pixels)
            self.assertIsNotNone(result.occurrences[0].context_image_path)
            self.assertTrue(result.occurrences[0].context_image_path.exists())
            self.assertEqual(1, len(result.tracks))
            self.assertEqual(1, len(result.keyframes))
            self.assertTrue(result.report.tex_path.exists())
            self.assertTrue((result.run_directory / "inventory" / "manifest.json").exists())
            self.assertTrue((result.run_directory / "inventory" / "occurrences.csv").exists())
            self.assertTrue((result.run_directory / "inventory" / "tracks.csv").exists())
            self.assertTrue((result.run_directory / "inventory" / "keyframes.csv").exists())
            self.assertTrue((result.run_directory / "inventory" / "media_info.json").exists())
            self.assertTrue((result.run_directory / "inventory" / "search.json").exists())

    def test_pipeline_discards_faces_below_minimum_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            self._create_test_image(image_path)

            config = self._config()
            config = AppConfig(
                app=config.app,
                media=config.media,
                video=config.video,
                face_model=FaceModelSettings(
                    backend=config.face_model.backend,
                    model_name=config.face_model.model_name,
                    det_size=config.face_model.det_size,
                    minimum_face_quality=0.99,
                    ctx_id=config.face_model.ctx_id,
                    providers=config.face_model.providers,
                ),
                clustering=config.clustering,
                reporting=config.reporting,
                forensics=config.forensics,
            )

            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            result = service.run(root)

            self.assertEqual(0, result.summary.total_occurrences)
            self.assertEqual(0, result.summary.total_clusters)
            self.assertEqual(1, result.summary.total_detected_face_sizes.count)
            self.assertEqual(0, result.summary.selected_face_sizes.count)

    def test_pipeline_discards_faces_below_minimum_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            self._create_test_image(image_path)

            config = self._config()
            config = AppConfig(
                app=config.app,
                media=config.media,
                video=config.video,
                face_model=FaceModelSettings(
                    backend=config.face_model.backend,
                    model_name=config.face_model.model_name,
                    det_size=config.face_model.det_size,
                    minimum_face_quality=config.face_model.minimum_face_quality,
                    minimum_face_size_pixels=50,
                    ctx_id=config.face_model.ctx_id,
                    providers=config.face_model.providers,
                ),
                clustering=config.clustering,
                reporting=config.reporting,
                forensics=config.forensics,
            )

            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            result = service.run(root)

            self.assertEqual(0, result.summary.total_occurrences)
            self.assertEqual(0, result.summary.total_clusters)
            self.assertEqual(1, result.summary.total_detected_face_sizes.count)
            self.assertEqual(0, result.summary.selected_face_sizes.count)

    def test_pipeline_assigns_unique_ids_across_multiple_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_test_image(root / "face_1.jpg")
            self._create_test_image(root / "face_2.jpg")

            service = InventoryService(
                config=self._config(),
                scanner_service=ScannerService(self._config().media),
                hashing_service=HashingService(),
                media_service=VideoService(self._config().video),
                clustering_service=ClusteringService(self._config().clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            result = service.run(root)

            occurrence_ids = [occurrence.occurrence_id for occurrence in result.occurrences]
            track_ids = [track.track_id for track in result.tracks]
            keyframe_ids = [keyframe.keyframe_id for keyframe in result.keyframes]

            self.assertEqual(len(occurrence_ids), len(set(occurrence_ids)))
            self.assertEqual(len(track_ids), len(set(track_ids)))
            self.assertEqual(len(keyframe_ids), len(set(keyframe_ids)))

    def test_list_planned_files_excludes_output_directory_and_keeps_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_test_image(root / "face.jpg")
            (root / "notes.txt").write_text("irrelevante", encoding="utf-8")
            output_dir = root / self._config().app.output_directory_name / "run_old"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._create_test_image(output_dir / "ignored.jpg")

            service = InventoryService(
                config=self._config(),
                scanner_service=ScannerService(self._config().media),
                hashing_service=HashingService(),
                media_service=VideoService(self._config().video),
                clustering_service=ClusteringService(self._config().clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            planned_files = service.list_planned_files(root)

            self.assertEqual(
                [
                    (root / "face.jpg").resolve(),
                    (root / "notes.txt").resolve(),
                ],
                [path for path, _ in planned_files],
            )
            self.assertEqual(["IMAGE", "OTHER"], [media_type.name for _, media_type in planned_files])

    def test_face_search_pipeline_generates_search_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "acervo"
            root.mkdir(parents=True, exist_ok=True)
            self._create_test_image(root / "face_1.jpg")
            self._create_test_image(root / "face_2.jpg")
            query_image = temp_root / "consulta.jpg"
            self._create_test_image(query_image)

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
                face_search_report_generator=_FakeFaceSearchReportGenerator(),
            )

            result = service.run_face_search(root, query_image)

            self.assertEqual(query_image.resolve(), result.query.source_path)
            self.assertGreaterEqual(len(result.matches), 1)
            self.assertEqual(1.0, result.matches[0].track_score)
            self.assertTrue(result.report.tex_path.exists())
            self.assertTrue(result.report.docx_path.exists())
            self.assertTrue((result.inventory_result.run_directory / "inventory" / "face_search.json").exists())

    def test_face_search_pipeline_accepts_multiple_query_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "acervo"
            root.mkdir(parents=True, exist_ok=True)
            self._create_test_image(root / "face_1.jpg")
            self._create_test_image(root / "face_2.jpg")
            query_images = [
                temp_root / "consulta_1.jpg",
                temp_root / "consulta_2.jpg",
            ]
            for query_image in query_images:
                self._create_test_image(query_image)

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
                face_search_report_generator=_FakeFaceSearchReportGenerator(),
            )

            result = service.run_face_search(root, query_images)

            self.assertEqual(2, result.summary.query_image_count)
            self.assertEqual(2, result.summary.query_faces_selected)
            self.assertEqual(0, result.summary.query_images_rejected)
            self.assertEqual(2, len(result.queries))
            self.assertEqual(query_images[0].resolve(), result.query.source_path)
            self.assertEqual(
                {path.resolve() for path in query_images},
                {query.source_path for query in result.queries},
            )
            self.assertEqual(2, len(result.query_events))
            self.assertTrue(all(event.status == "selected" for event in result.query_events))
            self.assertGreaterEqual(len(result.matches), 1)
            self.assertIn(
                result.matches[0].query_source_path,
                {path.resolve() for path in query_images},
            )
            self.assertTrue(result.report.tex_path.exists())
            self.assertTrue(result.report.docx_path.exists())
            self.assertTrue((result.inventory_result.run_directory / "inventory" / "face_search.json").exists())

    def test_face_search_pipeline_reports_rejected_query_files_for_chain_of_custody(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "acervo"
            root.mkdir(parents=True, exist_ok=True)
            self._create_test_image(root / "face_1.jpg")
            valid_query = temp_root / "consulta_valida.jpg"
            invalid_query = temp_root / "consulta_corrompida.jpg"
            self._create_test_image(valid_query)
            invalid_query.write_text("arquivo corrompido", encoding="utf-8")

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
                face_search_report_generator=_FakeFaceSearchReportGenerator(),
            )

            result = service.run_face_search(root, [valid_query, invalid_query])

            self.assertEqual(2, result.summary.query_image_count)
            self.assertEqual(1, result.summary.query_faces_selected)
            self.assertEqual(1, result.summary.query_images_rejected)
            self.assertEqual(2, len(result.query_events))
            rejected = next(event for event in result.query_events if event.status == "rejected")
            self.assertEqual(invalid_query.resolve(), rejected.source_path)
            self.assertEqual("MediaDecodeError", rejected.error_type)
            self.assertIsNotNone(rejected.error_message)
            self.assertIn("Nao foi possivel ler a imagem", rejected.error_message)
            exported = (result.inventory_result.run_directory / "inventory" / "face_search.json").read_text(encoding="utf-8")
            self.assertIn("query_events", exported)
            self.assertIn("consulta_corrompida.jpg", exported)

    def test_face_search_pipeline_generates_auditable_report_when_all_queries_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            root = temp_root / "acervo"
            root.mkdir(parents=True, exist_ok=True)
            self._create_test_image(root / "face_1.jpg")
            invalid_query = temp_root / "consulta_corrompida.jpg"
            invalid_query.write_text("arquivo corrompido", encoding="utf-8")

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
                face_search_report_generator=_FakeFaceSearchReportGenerator(),
            )

            result = service.run_face_search(root, [invalid_query])

            self.assertIsNone(result.query)
            self.assertEqual([], result.queries)
            self.assertEqual(1, result.summary.query_image_count)
            self.assertEqual(0, result.summary.query_faces_selected)
            self.assertEqual(1, result.summary.query_images_rejected)
            self.assertEqual(0, len(result.matches))
            self.assertEqual(1, len(result.query_events))
            self.assertEqual("rejected", result.query_events[0].status)
            self.assertTrue(result.report.tex_path.exists())
            self.assertTrue(result.report.docx_path.exists())

    def test_face_set_comparison_exports_matches_and_mesh_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            set_a_image = temp_root / "set_a.jpg"
            set_b_image = temp_root / "set_b.jpg"
            self._create_test_image(set_a_image)
            self._create_test_image(set_b_image)

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            result = service.compare_face_sets([set_a_image], [set_b_image], work_directory=temp_root / "trabalho")

            self.assertEqual(1, result.summary.set_a_images)
            self.assertEqual(1, result.summary.set_b_images)
            self.assertEqual(1, result.summary.set_a_selected_faces)
            self.assertEqual(1, result.summary.set_b_selected_faces)
            self.assertEqual(1, result.summary.total_pair_comparisons)
            self.assertEqual(1, result.summary.assignment_matches)
            self.assertEqual(0, result.summary.candidate_matches)
            self.assertEqual(1, len(result.matches))
            self.assertEqual("assignment", result.matches[0].classification)
            self.assertEqual(1.0, result.matches[0].similarity)
            self.assertEqual(1.0, result.summary.q1_similarity)
            self.assertEqual(1.0, result.summary.q3_similarity)
            self.assertEqual(1.0, result.summary.mean_confidence_low)
            self.assertEqual(1.0, result.summary.mean_confidence_high)
            self.assertTrue(result.manifest_path.exists())
            self.assertTrue((result.export_directory / "face_set_comparison_matches.csv").exists())
            self.assertTrue((result.export_directory / "face_set_comparison_summary.txt").exists())
            self.assertTrue((result.export_directory / "inputs" / "set_a" / "0001_set_a.jpg").exists())
            self.assertTrue((result.export_directory / "inputs" / "set_b" / "0001_set_b.jpg").exists())
            self.assertEqual(1, len(result.set_a_faces))
            self.assertEqual(1, len(result.set_b_faces))
            self.assertIsNotNone(result.set_a_faces[0].mesh_crop_path)
            self.assertIsNotNone(result.set_a_faces[0].mesh_context_path)
            self.assertTrue(result.set_a_faces[0].mesh_crop_path.exists())
            self.assertTrue(result.set_a_faces[0].mesh_context_path.exists())
            self.assertGreaterEqual(len(result.set_a_faces[0].biometric_landmarks), 5)

    def test_face_set_comparison_can_calibrate_likelihood_ratio_with_labeled_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            set_a_image = temp_root / "query_a.jpg"
            set_b_image = temp_root / "query_b.jpg"
            self._create_test_image(set_a_image)
            self._create_test_image(set_b_image)

            calibration_root = temp_root / "calibracao"
            for identity_label in ("alice", "bob"):
                identity_root = calibration_root / identity_label
                identity_root.mkdir(parents=True, exist_ok=True)
                for index in range(1, 5):
                    self._create_test_image(identity_root / f"{identity_label}_{index}.jpg")

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_CalibrationAwareAnalyzer,
            )

            result = service.compare_face_sets(
                [set_a_image],
                [set_b_image],
                work_directory=temp_root / "trabalho",
                calibration_root=calibration_root,
            )

            self.assertIsNotNone(result.calibration)
            self.assertTrue(result.calibration.summary.support_ready)
            self.assertTrue(result.summary.likelihood_ratio_calibrated)
            self.assertEqual(1, result.summary.calibrated_matches)
            self.assertIsNotNone(result.matches[0].likelihood_ratio)
            self.assertIsNotNone(result.matches[0].log10_likelihood_ratio)
            self.assertGreater(result.matches[0].log10_likelihood_ratio, 0.0)
            self.assertIsNotNone(result.matches[0].evidence_label)
            self.assertTrue((result.export_directory / "face_set_comparison_calibration_scores.csv").exists())
            self.assertTrue((result.export_directory / "face_set_comparison_calibration_model.json").exists())

    def test_face_set_comparison_summary_text_includes_mann_whitney_quality_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            set_a_images = [temp_root / "set_a_1.jpg", temp_root / "set_a_2.jpg"]
            set_b_images = [temp_root / "set_b_1.jpg", temp_root / "set_b_2.jpg"]
            for path in [*set_a_images, *set_b_images]:
                self._create_test_image(path)

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            result = service.compare_face_sets(
                set_a_images,
                set_b_images,
                work_directory=temp_root / "trabalho",
            )

            summary_text = (result.export_directory / "face_set_comparison_summary.txt").read_text(encoding="utf-8")
            self.assertIn("Teste nao parametrico entre grupos:", summary_text)
            self.assertIn("U de Mann-Whitney bilateral sobre qualidade facial", summary_text)
            self.assertIn("p-valor bilateral:", summary_text)
            self.assertIn("n: Padrao 2 | Questionado 2", summary_text)

    def test_face_set_comparison_can_reuse_saved_likelihood_ratio_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            set_a_image = temp_root / "query_a.jpg"
            set_b_image = temp_root / "query_b.jpg"
            self._create_test_image(set_a_image)
            self._create_test_image(set_b_image)

            calibration_root = temp_root / "calibracao"
            for identity_label in ("alice", "bob"):
                identity_root = calibration_root / identity_label
                identity_root.mkdir(parents=True, exist_ok=True)
                for index in range(1, 5):
                    self._create_test_image(identity_root / f"{identity_label}_{index}.jpg")

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_CalibrationAwareAnalyzer,
            )

            calibrated_result = service.compare_face_sets(
                [set_a_image],
                [set_b_image],
                work_directory=temp_root / "trabalho_base",
                calibration_root=calibration_root,
            )
            model_path = calibrated_result.export_directory / "face_set_comparison_calibration_model.json"

            reused_result = service.compare_face_sets(
                [set_a_image],
                [set_b_image],
                work_directory=temp_root / "trabalho_reuso",
                calibration_model_path=model_path,
            )

            self.assertTrue(model_path.exists())
            self.assertIsNotNone(reused_result.calibration)
            self.assertTrue(reused_result.calibration.loaded_from_model)
            self.assertEqual(model_path.resolve(), reused_result.calibration.model_path)
            self.assertEqual([], reused_result.calibration.inputs)
            self.assertEqual([], reused_result.calibration.entries)
            self.assertTrue(reused_result.calibration.summary.support_ready)
            self.assertTrue(reused_result.summary.likelihood_ratio_calibrated)
            self.assertAlmostEqual(
                calibrated_result.matches[0].log10_likelihood_ratio,
                reused_result.matches[0].log10_likelihood_ratio,
                places=6,
            )
            self.assertFalse((reused_result.export_directory / "face_set_comparison_calibration_inputs.csv").exists())
            self.assertFalse((reused_result.export_directory / "face_set_comparison_calibration_entries.csv").exists())
            self.assertTrue((reused_result.export_directory / "face_set_comparison_calibration_model.json").exists())

    def test_face_set_comparison_can_migrate_saved_likelihood_ratio_model_without_reprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            set_a_image = temp_root / "query_a.jpg"
            set_b_image = temp_root / "query_b.jpg"
            self._create_test_image(set_a_image)
            self._create_test_image(set_b_image)

            calibration_root = temp_root / "calibracao"
            for identity_label in ("alice", "bob"):
                identity_root = calibration_root / identity_label
                identity_root.mkdir(parents=True, exist_ok=True)
                for index in range(1, 5):
                    self._create_test_image(identity_root / f"{identity_label}_{index}.jpg")

            base_config = self._config()
            legacy_config = replace(
                base_config,
                likelihood_ratio=replace(base_config.likelihood_ratio, density_estimator="gaussian_kde"),
            )
            legacy_service = InventoryService(
                config=legacy_config,
                scanner_service=ScannerService(legacy_config.media),
                hashing_service=HashingService(),
                media_service=VideoService(legacy_config.video),
                clustering_service=ClusteringService(legacy_config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_CalibrationAwareAnalyzer,
            )

            legacy_result = legacy_service.compare_face_sets(
                [set_a_image],
                [set_b_image],
                work_directory=temp_root / "trabalho_legado",
                calibration_root=calibration_root,
            )
            legacy_model_path = legacy_result.export_directory / "face_set_comparison_calibration_model.json"

            current_service = InventoryService(
                config=base_config,
                scanner_service=ScannerService(base_config.media),
                hashing_service=HashingService(),
                media_service=VideoService(base_config.video),
                clustering_service=ClusteringService(base_config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_CalibrationAwareAnalyzer,
            )
            migrated_model_path = current_service.migrate_face_set_comparison_calibration_model(
                legacy_model_path,
                temp_root / "modelo_migrado.json",
            )
            migrated_model = current_service.load_face_set_comparison_calibration_model(migrated_model_path)

            self.assertEqual("gaussian_kde", legacy_result.calibration.summary.density_method)
            self.assertEqual("bounded_logit_kde", migrated_model.summary.density_method)
            self.assertIsNotNone(migrated_model.settings_snapshot)
            self.assertEqual("bounded_logit_kde", migrated_model.settings_snapshot.density_estimator)
            self.assertEqual(legacy_result.calibration.genuine_scores, migrated_model.genuine_scores)
            self.assertEqual(legacy_result.calibration.impostor_scores, migrated_model.impostor_scores)
            self.assertTrue(
                any(
                    "Modelo migrado sem reprocessar imagens" in line
                    for line in migrated_model.procedure_details
                )
            )

            reused_result = current_service.compare_face_sets(
                [set_a_image],
                [set_b_image],
                work_directory=temp_root / "trabalho_migrado",
                calibration_model_path=migrated_model_path,
            )

            self.assertTrue(reused_result.summary.likelihood_ratio_calibrated)
            self.assertIsNotNone(reused_result.matches[0].likelihood_ratio)
            self.assertIsNotNone(reused_result.matches[0].log10_likelihood_ratio)

    def test_face_set_comparison_emits_phase_logs_for_calibration_and_matching(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            set_a_image = temp_root / "query_a.jpg"
            set_b_image = temp_root / "query_b.jpg"
            self._create_test_image(set_a_image)
            self._create_test_image(set_b_image)

            calibration_root = temp_root / "calibracao"
            for identity_label in ("alice", "bob"):
                identity_root = calibration_root / identity_label
                identity_root.mkdir(parents=True, exist_ok=True)
                for index in range(1, 5):
                    self._create_test_image(identity_root / f"{identity_label}_{index}.jpg")

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_CalibrationAwareAnalyzer,
            )
            logs: list[str] = []
            progress_messages: list[str] = []

            service.compare_face_sets(
                [set_a_image],
                [set_b_image],
                work_directory=temp_root / "trabalho",
                calibration_root=calibration_root,
                log_callback=logs.append,
                progress_callback=lambda current, total, message: progress_messages.append(message),
            )

            self.assertTrue(any("[Comparacao] Similaridades |" in line for line in logs))
            self.assertTrue(any("[Comparacao] Similaridades concluidas | pares=1 | ranking=1" in line for line in logs))
            self.assertTrue(
                any(
                    "[Calibracao LR] Pares previstos | mesma_origem=12 | origem_distinta=16 | limite_amostra=20000"
                    in line
                    for line in logs
                )
            )
            self.assertTrue(
                any("[Calibracao LR] Mesma origem concluida | pares=12/12 | amostrados=12" in line for line in logs)
            )
            self.assertTrue(
                any(
                    "[Calibracao LR] Origem distinta concluida | pares=16/16 | amostrados=16" in line
                    for line in logs
                )
            )
            self.assertTrue(
                any("[Calibracao LR] Aplicacao de LR concluida | pares_calibrados=1" in line for line in logs)
            )
            self.assertTrue(any("Calibracao LR: preparando pares estatisticos" in message for message in progress_messages))
            self.assertTrue(
                any("Calibracao LR: ajustando densidades" in message for message in progress_messages)
            )

    def test_face_set_comparison_uses_custom_likelihood_ratio_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            set_a_image = temp_root / "query_a.jpg"
            set_b_image = temp_root / "query_b.jpg"
            self._create_test_image(set_a_image)
            self._create_test_image(set_b_image)

            calibration_root = temp_root / "calibracao"
            for identity_label in ("alice", "bob"):
                identity_root = calibration_root / identity_label
                identity_root.mkdir(parents=True, exist_ok=True)
                for index in range(1, 5):
                    self._create_test_image(identity_root / f"{identity_label}_{index}.jpg")

            base_config = self._config()
            config = AppConfig(
                app=base_config.app,
                media=base_config.media,
                video=base_config.video,
                face_model=base_config.face_model,
                clustering=base_config.clustering,
                reporting=base_config.reporting,
                forensics=base_config.forensics,
                tracking=base_config.tracking,
                enhancement=base_config.enhancement,
                search=base_config.search,
                distributed=base_config.distributed,
                likelihood_ratio=LikelihoodRatioSettings(
                    max_scores_per_distribution=3,
                    minimum_identities_with_faces=2,
                    minimum_same_source_scores=3,
                    minimum_different_source_scores=3,
                    minimum_unique_scores_per_distribution=2,
                    density_estimator="gaussian_kde",
                    kde_bandwidth_scale=1.3,
                    kde_uniform_floor_weight=0.01,
                    kde_min_density=1e-9,
                ),
            )
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_CalibrationAwareAnalyzer,
            )

            result = service.compare_face_sets(
                [set_a_image],
                [set_b_image],
                work_directory=temp_root / "trabalho",
                calibration_root=calibration_root,
            )

            self.assertIsNotNone(result.calibration)
            self.assertEqual(3, result.calibration.summary.genuine_score_count)
            self.assertEqual(3, result.calibration.summary.impostor_score_count)
            self.assertIsNotNone(result.calibration.settings_snapshot)
            self.assertEqual("gaussian_kde", result.calibration.summary.density_method)
            self.assertEqual("gaussian_kde", result.calibration.settings_snapshot.density_estimator)
            self.assertEqual(1.3, result.calibration.settings_snapshot.kde_bandwidth_scale)
            self.assertEqual(0.01, result.calibration.settings_snapshot.kde_uniform_floor_weight)
            self.assertEqual(1e-9, result.calibration.settings_snapshot.kde_min_density)
            self.assertIn("1.0000%", result.calibration.summary.smoothing_note or "")

    def test_pipeline_emits_detailed_and_accurate_video_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "amostra.mp4"
            video_path.write_bytes(b"fake-video")
            sampled_frames = [
                SampledFrame(
                    source_path=video_path,
                    image_name="frame_000060",
                    frame_index=60,
                    timestamp_seconds=2.0,
                    bgr_pixels=np.zeros((80, 80, 3), dtype=np.uint8),
                    original_bgr_pixels=np.zeros((80, 80, 3), dtype=np.uint8),
                ),
                SampledFrame(
                    source_path=video_path,
                    image_name="frame_000120",
                    frame_index=120,
                    timestamp_seconds=4.0,
                    bgr_pixels=np.zeros((80, 80, 3), dtype=np.uint8),
                    original_bgr_pixels=np.zeros((80, 80, 3), dtype=np.uint8),
                ),
            ]
            service = InventoryService(
                config=self._config(),
                scanner_service=ScannerService(self._config().media),
                hashing_service=HashingService(),
                media_service=_FakeVideoService(
                    self._config().video,
                    frames=sampled_frames,
                    sampling_info=VideoSamplingInfo(
                        fps=30.0,
                        total_frames=900,
                        duration_seconds=30.0,
                        frame_step=60,
                        actual_sampling_interval_seconds=2.0,
                        planned_sample_count=2,
                        max_sample_count=10,
                    ),
                ),
                clustering_service=ClusteringService(self._config().clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
                media_info_extractor=_FakeMediaInfoExtractor(),
            )
            logs: list[str] = []

            service.run(root, log_callback=logs.append)

            self.assertTrue(any("[Planejamento] Arquivos previstos para processamento: 1" in line for line in logs))
            self.assertTrue(any("[Planejamento 1/1] tipo=video | caminho=" in line for line in logs))
            self.assertTrue(any("[Configuracao] Midias |" in line for line in logs))
            self.assertTrue(any("Video | fps=30.00" in line for line in logs))
            self.assertTrue(any("quadro_real=000060" in line for line in logs))
            self.assertTrue(any("[Tracking] resumo | origem=amostra.mp4" in line for line in logs))
            self.assertTrue(any("[Arquivo 1/1] Midia analisada | amostras=2" in line for line in logs))
            self.assertTrue(any("[Configuracao] Aprimoramento | pre_processamento=sim" in line for line in logs))
            self.assertTrue(any("[Resumo] arquivos=1 | midias=1" in line for line in logs))

    def test_pipeline_can_write_outputs_to_external_work_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "evidencias"
            work = Path(temp_dir) / "trabalho"
            root.mkdir(parents=True, exist_ok=True)
            work.mkdir(parents=True, exist_ok=True)
            image_path = root / "face.jpg"
            self._create_test_image(image_path)

            config = self._config()
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=VideoService(config.video),
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            result = service.run(root, work_directory=work)

            self.assertTrue(str(result.run_directory).startswith(str((work / config.app.output_directory_name).resolve())))
            self.assertFalse((root / config.app.output_directory_name).exists())

    def test_pipeline_can_use_local_temporary_copy_without_losing_original_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            self._create_test_image(image_path)

            config = self._config()
            config = AppConfig(
                app=AppSettings(
                    name=config.app.name,
                    output_directory_name=config.app.output_directory_name,
                    report_title=config.app.report_title,
                    organization=config.app.organization,
                    log_level=config.app.log_level,
                    use_local_temp_copy=True,
                ),
                media=config.media,
                video=config.video,
                face_model=config.face_model,
                clustering=config.clustering,
                reporting=config.reporting,
                forensics=config.forensics,
            )
            recording_media_service = _RecordingImageVideoService(config.video)
            service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=recording_media_service,
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            result = service.run(root)

            self.assertEqual(1, len(recording_media_service.loaded_image_paths))
            self.assertNotEqual(image_path.resolve(), recording_media_service.loaded_image_paths[0].resolve())
            self.assertEqual(image_path.resolve(), result.files[0].path.resolve())
            self.assertEqual(image_path.resolve(), result.occurrences[0].source_path.resolve())

    def test_local_run_resumes_incomplete_execution_and_skips_successful_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "evidencias"
            work = Path(temp_dir) / "trabalho"
            root.mkdir(parents=True, exist_ok=True)
            work.mkdir(parents=True, exist_ok=True)
            first_image = root / "a_face.jpg"
            second_image = root / "b_face.jpg"
            self._create_test_image(first_image)
            self._create_test_image(second_image)

            config = self._config()
            crashing_media_service = _RecordingImageVideoService(config.video)
            first_service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=crashing_media_service,
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            original_process = first_service._process_file_bundle
            call_count = {"value": 0}

            def crashing_process(*args, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 2:
                    raise RuntimeError("falha abrupta")
                return original_process(*args, **kwargs)

            with mock.patch.object(first_service, "_process_file_bundle", side_effect=crashing_process):
                with self.assertRaises(RuntimeError):
                    first_service.run(root, work_directory=work)

            output_root = work / config.app.output_directory_name
            [run_directory] = list(output_root.glob("run_*"))

            resumed_media_service = _RecordingImageVideoService(config.video)
            resumed_service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=resumed_media_service,
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FakeAnalyzer,
            )

            result = resumed_service.run(root, work_directory=work)

            self.assertEqual(run_directory.resolve(), result.run_directory.resolve())
            self.assertEqual([second_image.resolve()], [path.resolve() for path in resumed_media_service.loaded_image_paths])
            self.assertEqual(2, result.summary.total_files)
            self.assertEqual(2, result.summary.total_occurrences)
            self.assertTrue(result.report.tex_path.exists())

    def test_local_run_retries_failed_files_on_next_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "evidencias"
            work = Path(temp_dir) / "trabalho"
            root.mkdir(parents=True, exist_ok=True)
            work.mkdir(parents=True, exist_ok=True)
            image_path = root / "face.jpg"
            self._create_test_image(image_path)

            _FailOnceAnalyzer.remaining_failures = 1
            config = self._config()

            first_media_service = _RecordingImageVideoService(config.video)
            first_service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=first_media_service,
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FailOnceAnalyzer,
            )

            first_result = first_service.run(root, work_directory=work)

            self.assertEqual(0, first_result.summary.total_occurrences)
            self.assertEqual(1, len(first_result.files))
            self.assertIsNotNone(first_result.files[0].processing_error)

            second_media_service = _RecordingImageVideoService(config.video)
            second_service = InventoryService(
                config=config,
                scanner_service=ScannerService(config.media),
                hashing_service=HashingService(),
                media_service=second_media_service,
                clustering_service=ClusteringService(config.clustering),
                report_generator=_FakeReportGenerator(),
                face_analyzer_factory=_FailOnceAnalyzer,
            )

            second_result = second_service.run(root, work_directory=work)

            self.assertEqual(first_result.run_directory.resolve(), second_result.run_directory.resolve())
            self.assertEqual([image_path.resolve()], [path.resolve() for path in second_media_service.loaded_image_paths])
            self.assertEqual(1, second_result.summary.total_occurrences)
            self.assertIsNone(second_result.files[0].processing_error)

    def _config(self) -> AppConfig:
        return AppConfig(
            app=AppSettings(
                name="Inventario Faces",
                output_directory_name="inventario_faces_output",
                report_title="Relatorio Teste",
                organization="Lab Teste",
                use_local_temp_copy=False,
            ),
            media=MediaSettings(
                image_extensions=(".jpg", ".png"),
                video_extensions=(".mp4", ".avi"),
            ),
            video=VideoSettings(
                sampling_interval_seconds=1.0,
                max_frames_per_video=10,
            ),
            face_model=FaceModelSettings(
                backend="fake",
                model_name="fake",
                det_size=(640, 640),
                minimum_face_quality=0.6,
                minimum_face_size_pixels=20,
            ),
            clustering=ClusteringSettings(
                assignment_similarity=0.5,
                candidate_similarity=0.4,
                min_cluster_size=1,
            ),
            reporting=ReportingSettings(
                compile_pdf=False,
            ),
            forensics=ForensicsSettings(
                chain_of_custody_note="Teste"
            ),
        )

    def _create_test_image(self, output_path: Path) -> None:
        canvas = np.zeros((80, 80, 3), dtype=np.uint8)
        canvas[5:40, 5:40] = (255, 255, 255)
        cv2.imwrite(str(output_path), canvas)


if __name__ == "__main__":
    unittest.main()
