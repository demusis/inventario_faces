"""Backend facial direto em ONNX Runtime, sem dependencia do pacote insightface.

Usa os mesmos arquivos de modelo do bundle InsightFace (ex.: ``buffalo_l``):
detector SCRFD (``det_*.onnx``), reconhecedor ArcFace (``w600k_*.onnx``) e,
quando presente, o modelo de 106 landmarks (``2d106det.onnx``). O pre e o
pos-processamento replicam o pipeline do insightface para que os embeddings
sejam numericamente equivalentes aos do backend original.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from inventario_faces.domain.config import FaceModelSettings
from inventario_faces.domain.entities import BoundingBox, DetectedFace, SampledFrame
from inventario_faces.infrastructure.face_analyzer_insight import (
    FaceAnalyzerInitializationError,
    GPU_EXECUTION_PROVIDERS,
    _ensure_local_model_bundle,
    _get_onnxruntime_module,
    _try_preload_onnxruntime_gpu_dlls,
    available_execution_providers,
    resolve_execution_providers,
)
from inventario_faces.utils.math_utils import l2_normalize

# Template canonico de alinhamento ArcFace (112x112), identico ao
# insightface.utils.face_align.arcface_dst.
ARCFACE_DESTINATION_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Mesmos limiares padrao do insightface.app.FaceAnalysis.
DETECTION_SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.4

_FEATURE_STRIDES = (8, 16, 32)
_ANCHORS_PER_CELL = 2
_DETECTION_INPUT_MEAN = 127.5
_DETECTION_INPUT_STD = 128.0
_RECOGNITION_INPUT_MEAN = 127.5
_RECOGNITION_INPUT_STD = 127.5
_RECOGNITION_IMAGE_SIZE = 112
_LANDMARK_INPUT_SIZE = 192
_LANDMARK_COUNT = 106


def estimate_similarity_transform(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Estima a transformacao de similaridade 2D (Umeyama) que leva source em destination.

    Retorna a matriz 2x3 pronta para ``cv2.warpAffine``. Equivalente ao
    ``skimage.transform.SimilarityTransform.estimate`` usado pelo insightface.
    """

    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(destination, dtype=np.float64)
    point_count = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = dst_centered.T @ src_centered / point_count
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)

    sign_correction = np.ones(2, dtype=np.float64)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0:
        sign_correction[-1] = -1.0
    rotation = u_matrix @ np.diag(sign_correction) @ vt_matrix

    source_variance = (src_centered ** 2).sum() / point_count
    if source_variance <= 0.0:
        raise ValueError("Landmarks de origem degenerados para a transformacao de similaridade.")
    scale = float((singular_values * sign_correction).sum() / source_variance)

    translation = dst_mean - scale * (rotation @ src_mean)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:2, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    predictions = []
    for index in range(0, distance.shape[1], 2):
        px = points[:, index % 2] + distance[:, index]
        py = points[:, index % 2 + 1] + distance[:, index + 1]
        predictions.append(px)
        predictions.append(py)
    return np.stack(predictions, axis=-1)


def _nms(detections: np.ndarray, iou_threshold: float) -> list[int]:
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        xx1 = np.maximum(x1[best], x1[order[1:]])
        yy1 = np.maximum(y1[best], y1[order[1:]])
        xx2 = np.minimum(x2[best], x2[order[1:]])
        yy2 = np.minimum(y2[best], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width * height
        overlap = intersection / (areas[best] + areas[order[1:]] - intersection)
        order = order[1:][overlap <= iou_threshold]
    return keep


def select_model_files(model_directory: Path) -> tuple[Path, Path, Path | None]:
    """Localiza detector, reconhecedor e (opcional) modelo de landmarks no bundle."""

    onnx_files = sorted(model_directory.glob("*.onnx"))
    if not onnx_files:
        raise FaceAnalyzerInitializationError(
            f"Nenhum arquivo .onnx encontrado em {model_directory}."
        )

    def _first_match(*fragments: str) -> Path | None:
        for candidate in onnx_files:
            name = candidate.name.lower()
            if any(fragment in name for fragment in fragments):
                return candidate
        return None

    detector_path = _first_match("det_", "scrfd")
    recognizer_path = _first_match("w600k", "glintr", "arcface", "_r50", "_r100")
    landmark_path = _first_match("2d106")

    if detector_path is None or recognizer_path is None:
        raise FaceAnalyzerInitializationError(
            (
                f"Bundle de modelos incompleto em {model_directory}: "
                f"detector={'ok' if detector_path else 'ausente'}, "
                f"reconhecedor={'ok' if recognizer_path else 'ausente'}."
            )
        )
    return detector_path, recognizer_path, landmark_path


@dataclass
class OnnxFaceAnalyzer:
    """Analisador facial compativel com o protocolo FaceAnalyzer usando ONNX puro."""

    settings: FaceModelSettings

    def __post_init__(self) -> None:
        _try_preload_onnxruntime_gpu_dlls()
        ort = _get_onnxruntime_module()
        available = available_execution_providers()
        providers = resolve_execution_providers(self.settings.providers, available)

        model_directory = _ensure_local_model_bundle(self.settings.model_name)
        detector_path, recognizer_path, landmark_path = select_model_files(model_directory)
        try:
            self._detector_session = ort.InferenceSession(str(detector_path), providers=providers)
            self._recognizer_session = ort.InferenceSession(str(recognizer_path), providers=providers)
            self._landmark_session = (
                ort.InferenceSession(str(landmark_path), providers=providers)
                if landmark_path is not None
                else None
            )
        except Exception as exc:
            raise FaceAnalyzerInitializationError(
                f"Falha ao carregar modelos ONNX de {model_directory}: {exc}"
            ) from exc

        self._detector_input_name = self._detector_session.get_inputs()[0].name
        self._detector_output_names = [output.name for output in self._detector_session.get_outputs()]
        expected_outputs = len(_FEATURE_STRIDES) * 3  # scores, bboxes e kps por stride
        if len(self._detector_output_names) != expected_outputs:
            raise FaceAnalyzerInitializationError(
                (
                    f"Detector {detector_path.name} possui {len(self._detector_output_names)} "
                    f"saidas; o backend onnx suporta apenas variantes SCRFD com keypoints "
                    f"({expected_outputs} saidas, ex.: det_10g do bundle buffalo_l)."
                )
            )
        self._recognizer_input_name = self._recognizer_session.get_inputs()[0].name
        if self._landmark_session is not None:
            self._landmark_input_name = self._landmark_session.get_inputs()[0].name
        self._providers = list(providers)
        self._available_providers = list(available)
        self._model_dir = model_directory

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    @property
    def available_providers(self) -> list[str]:
        return list(self._available_providers)

    @property
    def using_gpu(self) -> bool:
        return any(provider in GPU_EXECUTION_PROVIDERS for provider in self._providers)

    def detect(self, frame: SampledFrame) -> list[DetectedFace]:
        image = np.asarray(frame.bgr_pixels)
        bboxes, scores, keypoint_sets = self._detect_faces(image)
        crop_source = (
            np.asarray(frame.original_bgr_pixels)
            if frame.original_bgr_pixels is not None
            else image
        )

        detections: list[DetectedFace] = []
        for bbox, score, keypoints in zip(bboxes, scores, keypoint_sets):
            x1, y1, x2, y2 = (float(value) for value in bbox)
            landmarks = tuple((float(point[0]), float(point[1])) for point in keypoints)
            biometric_landmarks = self._dense_landmarks(image, bbox) or landmarks
            detections.append(
                DetectedFace(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    detection_score=float(score),
                    crop_bgr=self._crop(crop_source, x1, y1, x2, y2),
                    landmarks=landmarks,
                    biometric_landmarks=tuple(biometric_landmarks),
                    enhancement_metadata=frame.enhancement_metadata,
                )
            )
        return detections

    def embed(self, frame: SampledFrame, detection: DetectedFace, reason: str = "keyframe") -> list[float]:
        if detection.landmarks:
            landmarks = np.asarray(
                [[float(x), float(y)] for x, y in detection.landmarks],
                dtype=np.float32,
            )
        else:
            landmarks = self._fallback_landmarks(detection.bbox)

        image = np.asarray(frame.bgr_pixels)
        transform = estimate_similarity_transform(landmarks, ARCFACE_DESTINATION_LANDMARKS)
        aligned = cv2.warpAffine(
            image,
            transform,
            (_RECOGNITION_IMAGE_SIZE, _RECOGNITION_IMAGE_SIZE),
            borderValue=0.0,
        )
        blob = cv2.dnn.blobFromImage(
            aligned,
            1.0 / _RECOGNITION_INPUT_STD,
            (_RECOGNITION_IMAGE_SIZE, _RECOGNITION_IMAGE_SIZE),
            (_RECOGNITION_INPUT_MEAN,) * 3,
            swapRB=True,
        )
        raw_embedding = self._recognizer_session.run(
            None, {self._recognizer_input_name: blob}
        )[0].flatten()
        return l2_normalize(raw_embedding.tolist())

    def analyze(self, frame: SampledFrame) -> list[DetectedFace]:
        detections = self.detect(frame)
        enriched: list[DetectedFace] = []
        for detection in detections:
            enriched.append(
                DetectedFace(
                    bbox=detection.bbox,
                    detection_score=detection.detection_score,
                    crop_bgr=detection.crop_bgr,
                    embedding=self.embed(frame, detection, reason="full_analysis"),
                    landmarks=detection.landmarks,
                    biometric_landmarks=detection.biometric_landmarks,
                    quality_metrics=detection.quality_metrics,
                    enhancement_metadata=detection.enhancement_metadata,
                    embedding_source="full_analysis",
                )
            )
        return enriched

    def _detection_canvas_size(self, image: np.ndarray) -> tuple[int, int]:
        configured = self.settings.det_size
        if configured is not None:
            width, height = int(configured[0]), int(configured[1])
        else:
            height, width = image.shape[:2]
            width = max(32, int(width))
            height = max(32, int(height))
        # Dimensoes multiplas de 32 mantem a grade de anchors alinhada com as
        # saidas do SCRFD em todos os strides.
        width = int(math.ceil(width / 32) * 32)
        height = int(math.ceil(height / 32) * 32)
        return width, height

    def _detect_faces(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        canvas_width, canvas_height = self._detection_canvas_size(image)
        image_height, image_width = image.shape[:2]

        image_ratio = image_height / float(image_width)
        canvas_ratio = canvas_height / float(canvas_width)
        if image_ratio > canvas_ratio:
            new_height = canvas_height
            new_width = max(1, int(new_height / image_ratio))
        else:
            new_width = canvas_width
            new_height = max(1, int(new_width * image_ratio))
        detection_scale = new_height / float(image_height)

        resized = cv2.resize(image, (new_width, new_height))
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:new_height, :new_width, :] = resized

        scores_list, bboxes_list, kpss_list = self._forward_detector(canvas)
        if not scores_list:
            empty = np.zeros((0,), dtype=np.float32)
            return np.zeros((0, 4)), empty, np.zeros((0, 5, 2))

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list) / detection_scale
        keypoints = np.vstack(kpss_list) / detection_scale

        order = scores.argsort()[::-1]
        detections = np.hstack([bboxes, scores[:, None]]).astype(np.float32)[order]
        keypoints = keypoints[order]
        keep = _nms(detections, NMS_IOU_THRESHOLD)
        detections = detections[keep]
        keypoints = keypoints[keep]
        return detections[:, :4], detections[:, 4], keypoints.reshape((-1, 5, 2))

    def _forward_detector(
        self, canvas: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        canvas_height, canvas_width = canvas.shape[:2]
        blob = cv2.dnn.blobFromImage(
            canvas,
            1.0 / _DETECTION_INPUT_STD,
            (canvas_width, canvas_height),
            (_DETECTION_INPUT_MEAN,) * 3,
            swapRB=True,
        )
        outputs = self._detector_session.run(
            self._detector_output_names, {self._detector_input_name: blob}
        )
        batched = len(outputs[0].shape) == 3

        stride_count = len(_FEATURE_STRIDES)
        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []
        for index, stride in enumerate(_FEATURE_STRIDES):
            if batched:
                scores = outputs[index][0]
                bbox_predictions = outputs[index + stride_count][0] * stride
                kps_predictions = outputs[index + stride_count * 2][0] * stride
            else:
                scores = outputs[index]
                bbox_predictions = outputs[index + stride_count] * stride
                kps_predictions = outputs[index + stride_count * 2] * stride

            grid_height = canvas_height // stride
            grid_width = canvas_width // stride
            anchor_centers = np.stack(
                np.mgrid[:grid_height, :grid_width][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if _ANCHORS_PER_CELL > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * _ANCHORS_PER_CELL, axis=1
                ).reshape((-1, 2))

            flat_scores = scores.ravel()
            positive_indices = np.where(flat_scores >= DETECTION_SCORE_THRESHOLD)[0]
            if positive_indices.size == 0:
                continue
            bboxes = _distance2bbox(anchor_centers, bbox_predictions)
            keypoints = _distance2kps(anchor_centers, kps_predictions)
            scores_list.append(flat_scores[positive_indices][:, None])
            bboxes_list.append(bboxes[positive_indices])
            kpss_list.append(keypoints[positive_indices])
        return scores_list, bboxes_list, kpss_list

    def _dense_landmarks(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> tuple[tuple[float, float], ...] | None:
        if self._landmark_session is None:
            return None
        x1, y1, x2, y2 = (float(value) for value in bbox)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        scale = _LANDMARK_INPUT_SIZE / (max(width, height) * 1.5)
        transform = np.array(
            [
                [scale, 0.0, _LANDMARK_INPUT_SIZE / 2.0 - center_x * scale],
                [0.0, scale, _LANDMARK_INPUT_SIZE / 2.0 - center_y * scale],
            ],
            dtype=np.float64,
        )
        aligned = cv2.warpAffine(
            image, transform, (_LANDMARK_INPUT_SIZE, _LANDMARK_INPUT_SIZE), borderValue=0.0
        )
        blob = cv2.dnn.blobFromImage(
            aligned,
            1.0,
            (_LANDMARK_INPUT_SIZE, _LANDMARK_INPUT_SIZE),
            (0.0, 0.0, 0.0),
            swapRB=True,
        )
        prediction = self._landmark_session.run(
            None, {self._landmark_input_name: blob}
        )[0][0]
        points = prediction.reshape((-1, 2)).astype(np.float64)
        if points.shape[0] > _LANDMARK_COUNT:
            points = points[-_LANDMARK_COUNT:, :]
        points += 1.0
        points *= _LANDMARK_INPUT_SIZE // 2

        inverse = cv2.invertAffineTransform(transform)
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        restored = homogeneous @ inverse.T
        return tuple((float(point[0]), float(point[1])) for point in restored)

    def _crop(self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        height, width = image.shape[:2]
        left = max(0, int(x1))
        top = max(0, int(y1))
        right = min(width, int(x2))
        bottom = min(height, int(y2))
        if right <= left or bottom <= top:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return image[top:bottom, left:right].copy()

    def _fallback_landmarks(self, bbox: BoundingBox) -> np.ndarray:
        width = bbox.width
        height = bbox.height
        return np.asarray(
            [
                [bbox.x1 + width * 0.30, bbox.y1 + height * 0.38],
                [bbox.x1 + width * 0.70, bbox.y1 + height * 0.38],
                [bbox.x1 + width * 0.50, bbox.y1 + height * 0.56],
                [bbox.x1 + width * 0.36, bbox.y1 + height * 0.78],
                [bbox.x1 + width * 0.64, bbox.y1 + height * 0.78],
            ],
            dtype=np.float32,
        )
