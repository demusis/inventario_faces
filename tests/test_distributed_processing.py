from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import (
    AppConfig,
    AppSettings,
    ClusteringSettings,
    DistributedSettings,
    FaceModelSettings,
    ForensicsSettings,
    MediaSettings,
    ReportingSettings,
    VideoSettings,
)
from inventario_faces.domain.entities import BoundingBox, DetectedFace, MediaType, ReportArtifacts, SampledFrame
from inventario_faces.infrastructure.distributed_coordination import DistributedCoordinator
from inventario_faces.services.clustering_service import ClusteringService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.inventory_service import InventoryService
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.video_service import VideoService


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
        tex_path.write_text("relatorio distribuido", encoding="utf-8")
        return ReportArtifacts(tex_path=tex_path, pdf_path=None, docx_path=None)


class _FakeMediaInfoExtractor:
    def extract(self, path: Path):
        from inventario_faces.domain.entities import MediaInfoAttribute, MediaInfoTrack

        return (
            (
                MediaInfoTrack(
                    track_type="Geral",
                    attributes=(MediaInfoAttribute(label="Formato", value="JPEG"),),
                ),
            ),
            None,
        )


class DistributedCoordinatorTests(unittest.TestCase):
    def test_claim_and_complete_updates_shared_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            image_path.write_bytes(b"abc")
            run_directory = root / "inventario_faces_output" / "cluster_compartilhado"
            coordinator = DistributedCoordinator(
                root_directory=root,
                run_directory=run_directory,
                settings=DistributedSettings(enabled=True, execution_label="compartilhado", node_name="node-a"),
            )

            [entry] = coordinator.load_or_create_plan([(image_path, MediaType.IMAGE)])
            claim_result = coordinator.try_claim(entry)

            self.assertEqual("claimed", claim_result.status)
            self.assertIsNotNone(claim_result.claim)

            partial_path = coordinator.write_partial_payload(
                entry,
                {
                    "file_record": {"path": str(image_path), "sha512": "a" * 128},
                    "occurrences": [],
                    "tracks": [],
                    "keyframes": [],
                    "raw_face_sizes": [],
                    "selected_face_sizes": [],
                },
                file_sha512="a" * 128,
            )
            coordinator.mark_completed(
                entry,
                partial_path=partial_path,
                sha512="a" * 128,
                occurrence_count=0,
                track_count=0,
                keyframe_count=0,
                processing_error=None,
            )
            coordinator.release_claim(claim_result.claim)

            snapshot = coordinator.snapshot(total_files=1)
            self.assertEqual(1, snapshot.completed_files)
            self.assertEqual(0, snapshot.active_claims)
            self.assertTrue(snapshot.is_complete)

            health = coordinator.inspect_health(total_files=1)
            self.assertEqual(1, health.healthy_partials)
            self.assertEqual(0, health.corrupted_partials)
            self.assertEqual(0, health.missing_partials)

    def test_stale_lock_is_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            image_path.write_bytes(b"abc")
            run_directory = root / "inventario_faces_output" / "cluster_compartilhado"
            coordinator = DistributedCoordinator(
                root_directory=root,
                run_directory=run_directory,
                settings=DistributedSettings(
                    enabled=True,
                    execution_label="compartilhado",
                    node_name="node-a",
                    stale_lock_timeout_minutes=1,
                ),
            )
            [entry] = coordinator.load_or_create_plan([(image_path, MediaType.IMAGE)])
            lock_path = coordinator.claims_directory / f"{entry.lock_stem}.lock"
            lock_path.write_text(
                json.dumps({"node_id": "node-b_999", "hostname": "node-b", "pid": 999}),
                encoding="utf-8",
            )
            old_time = os.path.getmtime(lock_path) - 7200
            os.utime(lock_path, (old_time, old_time))

            claim_result = coordinator.try_claim(entry)

            self.assertEqual("claimed", claim_result.status)
            self.assertIsNotNone(claim_result.claim)

    def test_corrupted_partial_is_flagged_in_health_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            image_path.write_bytes(b"abc")
            run_directory = root / "inventario_faces_output" / "cluster_compartilhado"
            coordinator = DistributedCoordinator(
                root_directory=root,
                run_directory=run_directory,
                settings=DistributedSettings(enabled=True, execution_label="compartilhado", node_name="node-a"),
            )

            [entry] = coordinator.load_or_create_plan([(image_path, MediaType.IMAGE)])
            partial_path = coordinator.partials_directory / f"{entry.lock_stem}.json"
            partial_path.write_text("{corrompido", encoding="utf-8")
            coordinator.mark_completed(
                entry,
                partial_path=partial_path,
                sha512="b" * 128,
                occurrence_count=0,
                track_count=0,
                keyframe_count=0,
                processing_error=None,
            )

            health = coordinator.inspect_health(total_files=1)

            self.assertEqual(0, health.healthy_partials)
            self.assertEqual(1, health.corrupted_partials)
            self.assertTrue(health.recovery_needed)

    def test_write_node_heartbeat_retries_transient_replace_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_directory = root / "inventario_faces_output" / "cluster_compartilhado"
            coordinator = DistributedCoordinator(
                root_directory=root,
                run_directory=run_directory,
                settings=DistributedSettings(enabled=True, execution_label="compartilhado", node_name="node-a"),
            )

            original_replace = os.replace
            call_count = {"value": 0}

            def flaky_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
                call_count["value"] += 1
                if call_count["value"] == 1:
                    raise PermissionError("arquivo temporariamente bloqueado")
                original_replace(source, target)

            with mock.patch("inventario_faces.infrastructure.distributed_coordination.os.replace", side_effect=flaky_replace):
                coordinator.write_node_heartbeat(total_files=1, phase="idle", current_entry=None)

            self.assertGreaterEqual(call_count["value"], 2)
            self.assertTrue(coordinator.node_file_path.exists())

    def test_write_node_heartbeat_does_not_abort_on_persistent_retryable_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_directory = root / "inventario_faces_output" / "cluster_compartilhado"
            coordinator = DistributedCoordinator(
                root_directory=root,
                run_directory=run_directory,
                settings=DistributedSettings(enabled=True, execution_label="compartilhado", node_name="node-a"),
            )

            with mock.patch(
                "inventario_faces.infrastructure.distributed_coordination.os.replace",
                side_effect=PermissionError("arquivo bloqueado"),
            ):
                coordinator.write_node_heartbeat(total_files=1, phase="idle", current_entry=None)

    def test_health_distinguishes_processable_files_from_total_scanned_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            note_path = root / "nota.txt"
            image_path.write_bytes(b"abc")
            note_path.write_text("irrelevante", encoding="utf-8")
            run_directory = root / "inventario_faces_output" / "cluster_compartilhado"
            coordinator = DistributedCoordinator(
                root_directory=root,
                run_directory=run_directory,
                settings=DistributedSettings(enabled=True, execution_label="compartilhado", node_name="node-a"),
            )

            entries = coordinator.load_or_create_plan(
                [
                    (image_path, MediaType.IMAGE),
                    (note_path, MediaType.OTHER),
                ]
            )

            health = coordinator.inspect_health(total_files=len(entries))
            snapshot = coordinator.snapshot(total_files=len(entries))

            self.assertEqual(2, health.total_files)
            self.assertEqual(1, health.processable_files)
            self.assertEqual(0, health.processable_completed_files)
            self.assertEqual(0, health.processable_active_claims)
            self.assertEqual(1, health.processable_pending_files)
            self.assertEqual(1, snapshot.processable_files)


class DistributedInventoryPipelineTests(unittest.TestCase):
    def test_distributed_run_generates_final_report_when_lote_is_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._create_test_image(root / "face.jpg")
            service = self._service(root, auto_finalize=True)

            result = service.run(root)

            self.assertIn("cluster_caso_compartilhado", str(result.run_directory))
            self.assertTrue(result.report.tex_path.exists())
            self.assertTrue((result.run_directory / "distributed" / "completed_manifest.json").exists())
            self.assertTrue((result.run_directory / "distributed" / "partials").exists())
            self.assertEqual(1, result.summary.total_files)
            self.assertEqual(1, result.summary.total_tracks)

    def test_distributed_run_with_external_work_directory_does_not_create_output_under_evidence_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "evidencias"
            work = base / "trabalho"
            root.mkdir(parents=True, exist_ok=True)
            work.mkdir(parents=True, exist_ok=True)
            self._create_test_image(root / "face.jpg")
            service = self._service(root, auto_finalize=True)

            result = service.run(root, work_directory=work)

            self.assertTrue(str(result.run_directory).startswith(str((work / "inventario_faces_output").resolve())))
            self.assertFalse((root / "inventario_faces_output").exists())

    def test_distributed_run_returns_status_file_when_item_is_claimed_by_other_node(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            self._create_test_image(image_path)

            config = self._config(auto_finalize=False)
            run_directory = root / config.app.output_directory_name / "cluster_caso_compartilhado"
            other_coordinator = DistributedCoordinator(
                root_directory=root,
                run_directory=run_directory,
                settings=replace(config.distributed, node_name="node-b"),
            )
            [entry] = other_coordinator.load_or_create_plan([(image_path.resolve(), MediaType.IMAGE)])
            claim_result = other_coordinator.try_claim(entry)
            self.assertEqual("claimed", claim_result.status)

            service = self._service(root, auto_finalize=False)
            result = service.run(root)

            self.assertEqual("progresso_distribuido.txt", result.report.tex_path.name)
            self.assertTrue(result.report.tex_path.exists())

    def test_distributed_finalization_recovers_corrupted_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "face.jpg"
            self._create_test_image(image_path)

            service = self._service(root, auto_finalize=True)
            initial_result = service.run(root)
            self.assertTrue(initial_result.report.tex_path.exists())

            run_directory = initial_result.run_directory
            partial_path = next((run_directory / "distributed" / "partials").glob("*.json"))
            partial_path.write_text("{corrompido", encoding="utf-8")

            health_result = service.inspect_distributed_health(root)
            self.assertEqual(1, health_result.health_snapshot.corrupted_partials)

            rerun_result = service.run(root)

            self.assertTrue(rerun_result.report.tex_path.exists())
            self.assertEqual(1, rerun_result.summary.total_files)
            self.assertEqual(1, rerun_result.summary.total_tracks)
            healed_health = service.inspect_distributed_health(root)
            self.assertEqual(0, healed_health.health_snapshot.corrupted_partials)
            self.assertEqual(1, healed_health.health_snapshot.healthy_partials)

    def _service(self, root: Path, auto_finalize: bool) -> InventoryService:
        config = self._config(auto_finalize=auto_finalize)
        return InventoryService(
            config=config,
            scanner_service=ScannerService(config.media),
            hashing_service=HashingService(),
            media_service=VideoService(config.video),
            clustering_service=ClusteringService(config.clustering),
            report_generator=_FakeReportGenerator(),
            face_analyzer_factory=_FakeAnalyzer,
            media_info_extractor=_FakeMediaInfoExtractor(),
        )

    def _config(self, auto_finalize: bool) -> AppConfig:
        return AppConfig(
            app=AppSettings(
                name="Inventario Faces",
                output_directory_name="inventario_faces_output",
                report_title="Relatorio Teste",
                organization="Lab Teste",
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
            distributed=DistributedSettings(
                enabled=True,
                execution_label="caso_compartilhado",
                node_name="node-a",
                heartbeat_interval_seconds=5,
                stale_lock_timeout_minutes=120,
                auto_finalize=auto_finalize,
                validate_partial_integrity=True,
                auto_reprocess_invalid_partials=True,
            ),
        )

    def _create_test_image(self, output_path: Path) -> None:
        canvas = np.zeros((80, 80, 3), dtype=np.uint8)
        canvas[5:40, 5:40] = (255, 255, 255)
        cv2.imwrite(str(output_path), canvas)


if __name__ == "__main__":
    unittest.main()
