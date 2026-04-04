from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    DetectedFace,
    FaceOccurrence,
    FaceTrack,
    KeyFrame,
    MediaType,
    SampledFrame,
    TrackQualityStatistics,
)
from inventario_faces.domain.protocols import FaceAnalyzer
from inventario_faces.infrastructure.artifact_store import ArtifactStore
from inventario_faces.utils.math_utils import (
    average_embeddings,
    bbox_iou,
    cosine_similarity,
    normalized_center_distance,
)
from inventario_faces.utils.time_utils import utc_now
from inventario_faces.utils.latex import format_seconds

from .enhancement_service import EnhancementService
from .quality_service import FaceQualityService

EventCallback = Callable[[str, dict[str, object]], None]
TextCallback = Callable[[str], None]


@dataclass(frozen=True)
class TrackingResult:
    occurrences: list[FaceOccurrence]
    tracks: list[FaceTrack]
    keyframes: list[KeyFrame]
    raw_face_sizes: tuple[float, ...]
    selected_face_sizes: tuple[float, ...]
    sampled_frames: int
    frames_with_faces: int
    raw_detection_count: int
    selected_detection_count: int
    embedded_detection_count: int


@dataclass
class _TrackState:
    track_id: str
    source_path: Path
    video_path: Path | None
    media_type: MediaType
    sha512: str
    start_frame: int | None
    end_frame: int | None
    start_time: float | None
    end_time: float | None
    last_bbox: object
    occurrence_ids: list[str] = field(default_factory=list)
    keyframe_ids: list[str] = field(default_factory=list)
    representative_embeddings: list[list[float]] = field(default_factory=list)
    detection_scores: list[float] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)
    sharpness_values: list[float] = field(default_factory=list)
    brightness_values: list[float] = field(default_factory=list)
    illumination_values: list[float] = field(default_factory=list)
    frontality_values: list[float] = field(default_factory=list)
    best_quality_score: float = 0.0
    best_occurrence_id: str | None = None
    preview_path: Path | None = None
    top_crops: list[tuple[float, Path]] = field(default_factory=list)
    last_keyframe_time: float | None = None
    last_keyframe_bbox: object | None = None
    last_keyframe_crop_histogram: np.ndarray | None = None
    last_keyframe_quality: float = 0.0
    missed_detections: int = 0

    @property
    def centroid_embedding(self) -> list[float]:
        return average_embeddings(self.representative_embeddings)

    def register_quality(self, score: float, sharpness: float, brightness: float, illumination: float, frontality: float, detection_score: float) -> None:
        self.detection_scores.append(detection_score)
        self.quality_scores.append(score)
        self.sharpness_values.append(sharpness)
        self.brightness_values.append(brightness)
        self.illumination_values.append(illumination)
        self.frontality_values.append(frontality)

    def add_top_crop(self, score: float, path: Path, max_items: int) -> None:
        self.top_crops.append((score, path))
        self.top_crops.sort(key=lambda item: item[0], reverse=True)
        del self.top_crops[max_items:]


@dataclass(frozen=True)
class _CandidateMatch:
    track: _TrackState
    geometry_score: float
    iou_score: float
    spatial_similarity: float
    embedding_similarity: float
    total_score: float
    required_embedding: bool


class FaceTrackingService:
    def __init__(
        self,
        config: AppConfig,
        enhancement_service: EnhancementService,
        quality_service: FaceQualityService,
    ) -> None:
        self._config = config
        self._enhancement_service = enhancement_service
        self._quality_service = quality_service

    def process_media(
        self,
        source_path: Path,
        sha512: str,
        media_type: MediaType,
        frames: Iterable[SampledFrame],
        analyzer: FaceAnalyzer,
        artifact_store: ArtifactStore,
        id_namespace: str = "",
        event_callback: EventCallback | None = None,
        text_callback: TextCallback | None = None,
    ) -> TrackingResult:
        active_tracks: list[_TrackState] = []
        completed_tracks: list[_TrackState] = []
        occurrences: list[FaceOccurrence] = []
        keyframes: list[KeyFrame] = []
        raw_face_sizes: list[float] = []
        selected_face_sizes: list[float] = []

        sampled_frames = 0
        frames_with_faces = 0
        raw_detection_count = 0
        selected_detection_count = 0
        embedded_detection_count = 0
        track_sequence = [0]

        for frame in frames:
            sampled_frames += 1
            enhanced_frame = self._enhancement_service.apply(frame)
            detections = self._detect_faces(analyzer, enhanced_frame)
            raw_detection_count += len(detections)
            raw_face_sizes.extend(
                min(detection.bbox.width, detection.bbox.height)
                for detection in detections
            )

            selected_detections = self._select_detections(detections)
            selected_detection_count += len(selected_detections)
            selected_face_sizes.extend(
                min(detection.bbox.width, detection.bbox.height)
                for detection in selected_detections
            )
            if selected_detections:
                frames_with_faces += 1

            enriched_detections = [
                self._enrich_detection(detection)
                for detection in selected_detections
            ]
            matched_tracks, embedded_count = self._match_detections(
                frame=enhanced_frame,
                analyzer=analyzer,
                detections=enriched_detections,
                active_tracks=active_tracks,
                track_sequence=track_sequence,
                id_namespace=id_namespace,
            )
            embedded_detection_count += embedded_count

            matched_track_ids = {track.track_id for _, track, _ in matched_tracks}
            for detection, track, used_embedding in matched_tracks:
                occurrence = self._create_occurrence(
                    detection=detection,
                    frame=enhanced_frame,
                    sha512=sha512,
                    media_type=media_type,
                    track=track,
                    occurrences=occurrences,
                    id_namespace=id_namespace,
                )
                track.occurrence_ids.append(occurrence.occurrence_id)
                track.end_frame = occurrence.frame_index
                track.end_time = occurrence.frame_timestamp_seconds
                track.last_bbox = occurrence.bbox
                track.missed_detections = 0
                self._update_track_statistics(track, occurrence)
                occurrences.append(occurrence)
                self._emit_event(
                    event_callback,
                    "track_detection_associated",
                    {
                        "track_id": track.track_id,
                        "occurrence_id": occurrence.occurrence_id,
                        "source_path": source_path,
                        "frame_index": occurrence.frame_index,
                        "frame_timestamp_seconds": occurrence.frame_timestamp_seconds,
                        "used_embedding_for_matching": used_embedding,
                    },
                )

                keyframe_reasons = self._keyframe_reasons(track, occurrence, detection)
                if keyframe_reasons:
                    if not occurrence.embedding:
                        occurrence.embedding = self._embed_detection(
                            analyzer,
                            enhanced_frame,
                            detection,
                            reason=",".join(keyframe_reasons),
                        )
                        occurrence.embedding_source = "keyframe"
                        embedded_detection_count += 1
                    keyframe = self._create_keyframe(
                        occurrence=occurrence,
                        detection=detection,
                        track=track,
                        frame=enhanced_frame,
                        artifact_store=artifact_store,
                        reasons=keyframe_reasons,
                        keyframes=keyframes,
                        id_namespace=id_namespace,
                    )
                    occurrence.is_keyframe = True
                    occurrence.keyframe_id = keyframe.keyframe_id
                    track.keyframe_ids.append(keyframe.keyframe_id)
                    if occurrence.embedding:
                        track.representative_embeddings.append(occurrence.embedding)
                        del track.representative_embeddings[
                            self._config.tracking.representative_embeddings_per_track :
                        ]
                    track.last_keyframe_time = occurrence.frame_timestamp_seconds
                    track.last_keyframe_bbox = occurrence.bbox
                    track.last_keyframe_crop_histogram = self._crop_histogram(detection.crop_bgr)
                    track.last_keyframe_quality = detection.quality_metrics.score if detection.quality_metrics else 0.0
                    keyframes.append(keyframe)
                    self._emit_event(
                        event_callback,
                        "keyframe_selected",
                        {
                            "track_id": track.track_id,
                            "occurrence_id": occurrence.occurrence_id,
                            "keyframe_id": keyframe.keyframe_id,
                            "reasons": list(keyframe_reasons),
                            "source_path": source_path,
                            "frame_index": occurrence.frame_index,
                        },
                    )

            for track in list(active_tracks):
                if track.track_id in matched_track_ids:
                    continue
                track.missed_detections += 1
                if track.missed_detections > self._config.tracking.max_missed_detections:
                    completed_tracks.append(track)
                    active_tracks.remove(track)
                    self._emit_event(
                        event_callback,
                        "track_closed",
                        {
                            "track_id": track.track_id,
                            "source_path": source_path,
                            "end_frame": track.end_frame,
                            "end_time": track.end_time,
                            "detections": len(track.occurrence_ids),
                        },
                    )

            new_tracks = [
                track
                for _, track, _ in matched_tracks
                if track not in active_tracks and track not in completed_tracks
            ]
            for track in new_tracks:
                active_tracks.append(track)

            self._emit_log(
                text_callback,
                (
                    f"[Tracking] {source_path.name} | amostra={sampled_frames} | "
                    f"{self._tracking_position_label(frame)} | "
                    f"deteccoes={len(detections)} | selecionadas={len(selected_detections)} | "
                    f"tracks_ativos={len(active_tracks)}"
                ),
            )

        completed_tracks.extend(active_tracks)
        return TrackingResult(
            occurrences=occurrences,
            tracks=[self._to_face_track(track) for track in completed_tracks],
            keyframes=keyframes,
            raw_face_sizes=tuple(raw_face_sizes),
            selected_face_sizes=tuple(selected_face_sizes),
            sampled_frames=sampled_frames,
            frames_with_faces=frames_with_faces,
            raw_detection_count=raw_detection_count,
            selected_detection_count=selected_detection_count,
            embedded_detection_count=embedded_detection_count,
        )

    def _select_detections(self, detections: list[DetectedFace]) -> list[DetectedFace]:
        min_quality = self._config.face_model.minimum_face_quality
        min_size = self._config.face_model.minimum_face_size_pixels
        return [
            detection
            for detection in detections
            if detection.detection_score >= min_quality
            and min(detection.bbox.width, detection.bbox.height) >= min_size
        ]

    def _enrich_detection(self, detection: DetectedFace) -> DetectedFace:
        quality_metrics = self._quality_service.assess(detection)
        return DetectedFace(
            bbox=detection.bbox,
            detection_score=detection.detection_score,
            crop_bgr=detection.crop_bgr,
            embedding=list(detection.embedding),
            landmarks=detection.landmarks,
            quality_metrics=quality_metrics,
            enhancement_metadata=detection.enhancement_metadata,
            embedding_source=detection.embedding_source,
        )

    def _match_detections(
        self,
        frame: SampledFrame,
        analyzer: FaceAnalyzer,
        detections: list[DetectedFace],
        active_tracks: list[_TrackState],
        track_sequence: list[int],
        id_namespace: str,
    ) -> tuple[list[tuple[DetectedFace, _TrackState, bool]], int]:
        matches: list[tuple[DetectedFace, _TrackState, bool]] = []
        used_tracks: set[str] = set()
        embedded_count = 0

        for detection in sorted(detections, key=lambda item: item.detection_score, reverse=True):
            candidates = self._candidate_matches(frame, detection, active_tracks, used_tracks)
            if any(candidate.required_embedding for candidate in candidates) and not detection.embedding:
                detection.embedding = self._embed_detection(analyzer, frame, detection, reason="association")
                detection.embedding_source = "association"
                embedded_count += 1
                candidates = self._candidate_matches(frame, detection, active_tracks, used_tracks)
            if not candidates:
                track = self._start_track(frame, detection, track_sequence, id_namespace)
                if not detection.embedding:
                    detection.embedding = self._embed_detection(analyzer, frame, detection, reason="track_start")
                    detection.embedding_source = "track_start"
                    embedded_count += 1
                matches.append((detection, track, bool(detection.embedding)))
                continue

            best = candidates[0]
            if len(candidates) > 1:
                margin = best.total_score - candidates[1].total_score
            else:
                margin = best.total_score

            if best.total_score < self._config.tracking.minimum_total_match_score or margin < self._config.tracking.confidence_margin:
                track = self._start_track(frame, detection, track_sequence, id_namespace)
                if not detection.embedding:
                    detection.embedding = self._embed_detection(analyzer, frame, detection, reason="track_start")
                    detection.embedding_source = "track_start"
                    embedded_count += 1
                matches.append((detection, track, bool(detection.embedding)))
                continue

            if best.required_embedding and not detection.embedding:
                detection.embedding = self._embed_detection(analyzer, frame, detection, reason="association")
                detection.embedding_source = "association"
                embedded_count += 1
            used_tracks.add(best.track.track_id)
            matches.append((detection, best.track, best.required_embedding))

        return matches, embedded_count

    def _candidate_matches(
        self,
        frame: SampledFrame,
        detection: DetectedFace,
        active_tracks: list[_TrackState],
        used_tracks: set[str],
    ) -> list[_CandidateMatch]:
        frame_height, frame_width = frame.bgr_pixels.shape[:2]
        candidates: list[_CandidateMatch] = []
        for track in active_tracks:
            if track.track_id in used_tracks:
                continue
            iou_score = bbox_iou(detection.bbox, track.last_bbox)
            distance = normalized_center_distance(
                detection.bbox,
                track.last_bbox,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            spatial_similarity = max(
                0.0,
                1.0 - (distance / max(self._config.tracking.spatial_distance_threshold, 1e-6)),
            )
            if iou_score < self._config.tracking.iou_threshold and distance > self._config.tracking.spatial_distance_threshold:
                continue
            geometry_score = (iou_score + spatial_similarity) / 2.0
            required_embedding = bool(track.representative_embeddings) and (
                geometry_score < 0.75 or len(active_tracks) > 1
            )
            embedding_similarity = 0.0
            if detection.embedding and track.representative_embeddings:
                embedding_similarity = cosine_similarity(detection.embedding, track.centroid_embedding)
                if embedding_similarity < self._config.tracking.embedding_similarity_threshold:
                    continue
            total_score = geometry_score
            if track.representative_embeddings and detection.embedding:
                total_score = (
                    self._config.tracking.geometry_weight * geometry_score
                    + self._config.tracking.embedding_weight * embedding_similarity
                )
            candidates.append(
                _CandidateMatch(
                    track=track,
                    geometry_score=geometry_score,
                    iou_score=iou_score,
                    spatial_similarity=spatial_similarity,
                    embedding_similarity=embedding_similarity,
                    total_score=total_score,
                    required_embedding=required_embedding,
                )
            )
        return sorted(candidates, key=lambda item: item.total_score, reverse=True)

    def _start_track(
        self,
        frame: SampledFrame,
        detection: DetectedFace,
        track_sequence: list[int],
        id_namespace: str,
    ) -> _TrackState:
        track_sequence[0] += 1
        track_id = self._scoped_id("T", track_sequence[0], id_namespace)
        return _TrackState(
            track_id=track_id,
            source_path=frame.source_path,
            video_path=frame.source_path if frame.frame_index is not None else None,
            media_type=MediaType.VIDEO if frame.frame_index is not None else MediaType.IMAGE,
            sha512="",
            start_frame=frame.frame_index,
            end_frame=frame.frame_index,
            start_time=frame.timestamp_seconds,
            end_time=frame.timestamp_seconds,
            last_bbox=detection.bbox,
        )

    def _create_occurrence(
        self,
        detection: DetectedFace,
        frame: SampledFrame,
        sha512: str,
        media_type: MediaType,
        track: _TrackState,
        occurrences: list[FaceOccurrence],
        id_namespace: str,
    ) -> FaceOccurrence:
        occurrence_id = self._scoped_id("O", len(occurrences) + 1, id_namespace)
        track.sha512 = sha512
        track.media_type = media_type
        return FaceOccurrence(
            occurrence_id=occurrence_id,
            source_path=frame.source_path,
            sha512=sha512,
            media_type=media_type,
            analysis_timestamp_utc=utc_now(),
            frame_index=frame.frame_index,
            frame_timestamp_seconds=frame.timestamp_seconds,
            bbox=detection.bbox,
            detection_score=detection.detection_score,
            crop_path=None,
            embedding=list(detection.embedding),
            context_image_path=None,
            track_id=track.track_id,
            quality_metrics=detection.quality_metrics,
            enhancement_metadata=detection.enhancement_metadata,
            track_position=len(track.occurrence_ids) + 1,
            embedding_source=detection.embedding_source,
        )

    def _update_track_statistics(self, track: _TrackState, occurrence: FaceOccurrence) -> None:
        quality = occurrence.quality_metrics
        if quality is None:
            return
        track.register_quality(
            score=quality.score,
            sharpness=quality.sharpness,
            brightness=quality.brightness,
            illumination=quality.illumination,
            frontality=quality.frontality,
            detection_score=occurrence.detection_score,
        )

    def _keyframe_reasons(
        self,
        track: _TrackState,
        occurrence: FaceOccurrence,
        detection: DetectedFace,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        quality_score = detection.quality_metrics.score if detection.quality_metrics is not None else 0.0
        if len(track.occurrence_ids) == 1:
            reasons.append("track_start")
        if track.last_keyframe_time is None:
            reasons.append("initial_reference")
        elif occurrence.frame_timestamp_seconds is not None:
            elapsed = occurrence.frame_timestamp_seconds - track.last_keyframe_time
            if elapsed >= self._config.video.keyframe_interval_seconds:
                reasons.append("interval")

        change_score = self._significant_change_score(track, detection)
        if change_score >= self._config.video.significant_change_threshold:
            reasons.append("significant_change")

        if quality_score >= track.best_quality_score + self._config.tracking.quality_improvement_margin:
            reasons.append("quality_peak")
        return tuple(dict.fromkeys(reasons))

    def _create_keyframe(
        self,
        occurrence: FaceOccurrence,
        detection: DetectedFace,
        track: _TrackState,
        frame: SampledFrame,
        artifact_store: ArtifactStore,
        reasons: tuple[str, ...],
        keyframes: list[KeyFrame],
        id_namespace: str,
    ) -> KeyFrame:
        keyframe_id = self._scoped_id("K", len(keyframes) + 1, id_namespace)
        crop_path = artifact_store.save_crop(occurrence.occurrence_id, detection.crop_bgr)
        source_image = frame.original_bgr_pixels if frame.original_bgr_pixels is not None else frame.bgr_pixels
        context_path = artifact_store.save_context(
            occurrence.occurrence_id,
            frame.image_name,
            source_image,
            detection.bbox,
        )
        occurrence.crop_path = crop_path
        occurrence.context_image_path = context_path

        quality_score = detection.quality_metrics.score if detection.quality_metrics else 0.0
        if quality_score >= track.best_quality_score:
            track.best_quality_score = quality_score
            track.best_occurrence_id = occurrence.occurrence_id
            track.preview_path = crop_path
        track.add_top_crop(quality_score, crop_path, self._config.tracking.top_crops_per_track)
        return KeyFrame(
            keyframe_id=keyframe_id,
            track_id=track.track_id,
            occurrence_id=occurrence.occurrence_id,
            source_path=occurrence.source_path,
            frame_index=occurrence.frame_index,
            timestamp_seconds=occurrence.frame_timestamp_seconds,
            selection_reasons=reasons,
            quality_metrics=detection.quality_metrics,
            detection_score=occurrence.detection_score,
            crop_path=crop_path,
            context_image_path=context_path,
            embedding=list(occurrence.embedding),
            preview_path=crop_path,
        )

    def _significant_change_score(self, track: _TrackState, detection: DetectedFace) -> float:
        if track.last_keyframe_bbox is None:
            return 0.0
        bbox_change = 1.0 - bbox_iou(detection.bbox, track.last_keyframe_bbox)
        histogram_change = 0.0
        current_histogram = self._crop_histogram(detection.crop_bgr)
        if track.last_keyframe_crop_histogram is not None and current_histogram is not None:
            histogram_change = 1.0 - float(
                cv2.compareHist(track.last_keyframe_crop_histogram, current_histogram, cv2.HISTCMP_CORREL)
            )
            histogram_change = max(0.0, min(histogram_change, 1.0))
        return max(bbox_change, histogram_change)

    def _crop_histogram(self, crop_bgr: object) -> np.ndarray | None:
        crop = np.asarray(crop_bgr)
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
        return cv2.normalize(histogram, histogram).flatten()

    def _scoped_id(self, prefix: str, sequence: int, namespace: str) -> str:
        normalized_namespace = "".join(character for character in namespace if character.isalnum())
        if normalized_namespace:
            return f"{prefix}{normalized_namespace}_{sequence:06d}"
        return f"{prefix}{sequence:06d}"

    def _to_face_track(self, track: _TrackState) -> FaceTrack:
        return FaceTrack(
            track_id=track.track_id,
            source_path=track.source_path,
            video_path=track.video_path,
            media_type=track.media_type,
            sha512=track.sha512,
            start_frame=track.start_frame,
            end_frame=track.end_frame,
            start_time=track.start_time,
            end_time=track.end_time,
            occurrence_ids=list(track.occurrence_ids),
            keyframe_ids=list(track.keyframe_ids),
            representative_embeddings=list(track.representative_embeddings),
            average_embedding=track.centroid_embedding,
            best_occurrence_id=track.best_occurrence_id,
            preview_path=track.preview_path,
            top_crop_paths=[path for _, path in track.top_crops],
            quality_statistics=self._track_quality_statistics(track),
        )

    def _track_quality_statistics(self, track: _TrackState) -> TrackQualityStatistics:
        def _mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        duration_seconds = 0.0
        if track.start_time is not None and track.end_time is not None:
            duration_seconds = max(0.0, track.end_time - track.start_time)
        return TrackQualityStatistics(
            total_detections=len(track.occurrence_ids),
            keyframe_count=len(track.keyframe_ids),
            mean_detection_score=_mean(track.detection_scores),
            max_detection_score=max(track.detection_scores) if track.detection_scores else 0.0,
            mean_quality_score=_mean(track.quality_scores),
            best_quality_score=max(track.quality_scores) if track.quality_scores else 0.0,
            mean_sharpness=_mean(track.sharpness_values),
            mean_brightness=_mean(track.brightness_values),
            mean_illumination=_mean(track.illumination_values),
            mean_frontality=_mean(track.frontality_values),
            duration_seconds=duration_seconds,
        )

    def _emit_event(self, callback: EventCallback | None, event: str, fields: dict[str, object]) -> None:
        if callback is not None:
            callback(event, fields)

    def _emit_log(self, callback: TextCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)

    def _tracking_position_label(self, frame: SampledFrame) -> str:
        if frame.frame_index is None:
            return "quadro_real=imagem_estatica | instante=-"
        return f"quadro_real={frame.frame_index:06d} | instante={format_seconds(frame.timestamp_seconds)}"

    def _detect_faces(self, analyzer: FaceAnalyzer, frame: SampledFrame) -> list[DetectedFace]:
        detect_method = getattr(analyzer, "detect", None)
        if callable(detect_method):
            return list(detect_method(frame))
        analyze_method = getattr(analyzer, "analyze", None)
        if callable(analyze_method):
            return list(analyze_method(frame))
        raise AttributeError("O analisador facial nao expoe detect() nem analyze().")

    def _embed_detection(
        self,
        analyzer: FaceAnalyzer,
        frame: SampledFrame,
        detection: DetectedFace,
        reason: str,
    ) -> list[float]:
        if detection.embedding:
            return list(detection.embedding)
        embed_method = getattr(analyzer, "embed", None)
        if callable(embed_method):
            return list(embed_method(frame, detection, reason=reason))
        return []
