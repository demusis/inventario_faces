from __future__ import annotations

import tempfile
import unittest
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
            )
        ]


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
