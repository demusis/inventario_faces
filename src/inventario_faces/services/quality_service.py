from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from inventario_faces.domain.entities import DetectedFace, FaceQualityMetrics


@dataclass(frozen=True)
class QualityWeights:
    detection: float = 0.40
    sharpness: float = 0.25
    frontality: float = 0.20
    illumination: float = 0.15


class FaceQualityService:
    def __init__(self, weights: QualityWeights | None = None) -> None:
        self._weights = weights or QualityWeights()

    def assess(self, detection: DetectedFace) -> FaceQualityMetrics:
        crop = np.asarray(detection.crop_bgr)
        if crop.size == 0:
            return FaceQualityMetrics(
                detection_score=detection.detection_score,
                bbox_pixels=min(detection.bbox.width, detection.bbox.height),
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = min(sharpness_raw / 500.0, 1.0)
        brightness = float(gray.mean() / 255.0)
        illumination = max(0.0, 1.0 - min(abs(brightness - 0.55) / 0.55, 1.0))
        frontality = self._frontality_score(detection)
        bbox_pixels = min(detection.bbox.width, detection.bbox.height)
        score = (
            self._weights.detection * detection.detection_score
            + self._weights.sharpness * sharpness
            + self._weights.frontality * frontality
            + self._weights.illumination * illumination
        )
        return FaceQualityMetrics(
            detection_score=detection.detection_score,
            sharpness=sharpness,
            brightness=brightness,
            illumination=illumination,
            frontality=frontality,
            bbox_pixels=bbox_pixels,
            score=max(0.0, min(score, 1.0)),
        )

    def _frontality_score(self, detection: DetectedFace) -> float:
        if len(detection.landmarks) < 5:
            return 0.5

        left_eye = np.asarray(detection.landmarks[0], dtype=np.float32)
        right_eye = np.asarray(detection.landmarks[1], dtype=np.float32)
        nose = np.asarray(detection.landmarks[2], dtype=np.float32)
        left_mouth = np.asarray(detection.landmarks[3], dtype=np.float32)
        right_mouth = np.asarray(detection.landmarks[4], dtype=np.float32)

        eye_distance = float(np.linalg.norm(left_eye - right_eye))
        if eye_distance == 0.0:
            return 0.5

        eye_center = (left_eye + right_eye) / 2.0
        nose_offset = abs(float(nose[0] - eye_center[0])) / max(eye_distance / 2.0, 1e-6)
        eye_tilt = abs(float(left_eye[1] - right_eye[1])) / eye_distance
        mouth_balance = abs(
            float(np.linalg.norm(nose - left_mouth) - np.linalg.norm(right_mouth - nose))
        ) / eye_distance

        penalties = [
            min(nose_offset, 1.0),
            min(eye_tilt * 2.0, 1.0),
            min(mouth_balance, 1.0),
        ]
        return max(0.0, 1.0 - (sum(penalties) / len(penalties)))
