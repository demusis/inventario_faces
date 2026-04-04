from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from inventario_faces.domain.config import EnhancementSettings
from inventario_faces.domain.entities import EnhancementMetadata, SampledFrame


@dataclass(frozen=True)
class EnhancementDecision:
    apply_clahe: bool
    apply_gamma: bool
    apply_denoise: bool


class EnhancementService:
    def __init__(self, settings: EnhancementSettings) -> None:
        self._settings = settings

    def apply(self, frame: SampledFrame) -> SampledFrame:
        original = np.asarray(frame.original_bgr_pixels if frame.original_bgr_pixels is not None else frame.bgr_pixels)
        brightness_before = self._brightness(original)
        decision = self._decision(brightness_before)
        if not self._settings.enable_preprocessing or not any(decision.__dict__.values()):
            metadata = EnhancementMetadata(
                applied=False,
                strategy="none",
                parameters={},
                brightness_before=brightness_before,
                brightness_after=brightness_before,
                note="Preprocessamento desabilitado ou desnecessario para este quadro.",
            )
            return SampledFrame(
                source_path=frame.source_path,
                image_name=frame.image_name,
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                bgr_pixels=original,
                original_bgr_pixels=original,
                enhancement_metadata=metadata,
            )

        enhanced = original.copy()
        parameters: dict[str, float | int | bool] = {}

        if decision.apply_clahe:
            enhanced = self._apply_clahe(enhanced)
            parameters["clahe_clip_limit"] = self._settings.clahe_clip_limit
            parameters["clahe_tile_grid_size"] = self._settings.clahe_tile_grid_size

        if decision.apply_gamma:
            enhanced = self._apply_gamma(enhanced, self._settings.gamma)
            parameters["gamma"] = self._settings.gamma

        if decision.apply_denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced,
                None,
                self._settings.denoise_strength,
                self._settings.denoise_strength,
                7,
                21,
            )
            parameters["denoise_strength"] = self._settings.denoise_strength

        brightness_after = self._brightness(enhanced)
        metadata = EnhancementMetadata(
            applied=True,
            strategy="clahe_gamma_denoise",
            parameters=parameters,
            brightness_before=brightness_before,
            brightness_after=brightness_after,
            note="Preprocessamento aplicado sem alterar o arquivo original.",
        )
        return SampledFrame(
            source_path=frame.source_path,
            image_name=frame.image_name,
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            bgr_pixels=enhanced,
            original_bgr_pixels=original,
            enhancement_metadata=metadata,
        )

    def _decision(self, brightness: float) -> EnhancementDecision:
        return EnhancementDecision(
            apply_clahe=brightness < self._settings.minimum_brightness_to_enhance,
            apply_gamma=abs(self._settings.gamma - 1.0) > 1e-6,
            apply_denoise=self._settings.denoise_strength > 0,
        )

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self._settings.clahe_clip_limit,
            tileGridSize=(self._settings.clahe_tile_grid_size, self._settings.clahe_tile_grid_size),
        )
        merged = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        if gamma <= 0.0:
            return image
        table = np.array(
            [((index / 255.0) ** (1.0 / gamma)) * 255.0 for index in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, table)

    def _brightness(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean() / 255.0)
