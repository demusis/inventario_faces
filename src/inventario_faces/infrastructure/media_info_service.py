from __future__ import annotations

import mimetypes
from pathlib import Path

import cv2
from PIL import Image, UnidentifiedImageError

from inventario_faces.domain.entities import MediaInfoAttribute, MediaInfoTrack


class MediaInfoService:
    def __init__(
        self,
        executable_path: str | None = None,
        directory: str | Path | None = None,
    ) -> None:
        self._legacy_executable_path = executable_path
        self._legacy_directory = Path(directory).expanduser().resolve() if directory not in (None, "") else None

    @property
    def executable_path(self) -> None:
        return None

    def extract(self, path: Path) -> tuple[tuple[MediaInfoTrack, ...], str | None]:
        candidate = Path(path)
        if not candidate.exists():
            return (), "Arquivo não encontrado para extração de metadados."

        image_tracks = self._extract_image_metadata(candidate)
        if image_tracks is not None:
            return image_tracks, None

        video_tracks, video_error = self._extract_video_metadata(candidate)
        if video_tracks is not None:
            return video_tracks, None
        return (), video_error

    def _extract_image_metadata(self, path: Path) -> tuple[MediaInfoTrack, ...] | None:
        try:
            with Image.open(path) as image:
                file_size = path.stat().st_size
                format_name = (image.format or path.suffix.lstrip(".") or "desconhecido").upper()
                mime = Image.MIME.get(image.format, mimetypes.guess_type(path.name)[0] or "desconhecido")
                compression = image.info.get("compression")
                image_track_attributes = [
                    MediaInfoAttribute("Formato", format_name),
                    MediaInfoAttribute("Tipo MIME", mime),
                    MediaInfoAttribute("Largura", f"{image.width} px"),
                    MediaInfoAttribute("Altura", f"{image.height} px"),
                    MediaInfoAttribute("Modo de cor", image.mode),
                    MediaInfoAttribute("Profundidade de bits", self._image_bit_depth(image.mode)),
                ]
                if compression:
                    image_track_attributes.append(MediaInfoAttribute("Compressão", str(compression)))
                return (
                    MediaInfoTrack(
                        track_type="Geral",
                        attributes=(
                            MediaInfoAttribute("Formato", format_name),
                            MediaInfoAttribute("Tipo MIME", mime),
                            MediaInfoAttribute("Tamanho do arquivo", self._format_file_size(file_size)),
                        ),
                    ),
                    MediaInfoTrack(
                        track_type="Imagem",
                        attributes=tuple(image_track_attributes),
                    ),
                )
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    def _extract_video_metadata(self, path: Path) -> tuple[tuple[MediaInfoTrack, ...] | None, str | None]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return None, "Não foi possível extrair metadados técnicos com o extrator interno."

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fourcc = self._decode_fourcc(int(capture.get(cv2.CAP_PROP_FOURCC) or 0))
            duration_seconds = frame_count / fps if fps > 0.0 and frame_count > 0 else 0.0
            file_size = path.stat().st_size
            bitrate = self._format_bitrate(file_size, duration_seconds)
            container_format = (path.suffix.lstrip(".") or "desconhecido").upper()
            mime = mimetypes.guess_type(path.name)[0] or "desconhecido"

            general_attributes = [
                MediaInfoAttribute("Formato", container_format),
                MediaInfoAttribute("Tipo MIME", mime),
                MediaInfoAttribute("Tamanho do arquivo", self._format_file_size(file_size)),
            ]
            if duration_seconds > 0:
                general_attributes.append(MediaInfoAttribute("Duração", self._format_duration(duration_seconds)))
            if bitrate is not None:
                general_attributes.append(MediaInfoAttribute("Taxa de bits global", bitrate))
            if frame_count > 0:
                general_attributes.append(MediaInfoAttribute("Número de quadros", str(frame_count)))

            video_attributes = [MediaInfoAttribute("Formato", container_format)]
            if fourcc:
                video_attributes.append(MediaInfoAttribute("Codec", fourcc))
            if width > 0:
                video_attributes.append(MediaInfoAttribute("Largura", f"{width} px"))
            if height > 0:
                video_attributes.append(MediaInfoAttribute("Altura", f"{height} px"))
            if width > 0 and height > 0:
                video_attributes.append(MediaInfoAttribute("Proporção de exibição", self._format_aspect_ratio(width, height)))
            if duration_seconds > 0:
                video_attributes.append(MediaInfoAttribute("Duração", self._format_duration(duration_seconds)))
            if bitrate is not None:
                video_attributes.append(MediaInfoAttribute("Taxa de bits", bitrate))
            if fps > 0:
                video_attributes.append(MediaInfoAttribute("Taxa de quadros", f"{fps:.3f} FPS"))
            if frame_count > 0:
                video_attributes.append(MediaInfoAttribute("Número de quadros", str(frame_count)))

            tracks = [
                MediaInfoTrack(track_type="Geral", attributes=tuple(general_attributes)),
            ]
            if len(video_attributes) > 1:
                tracks.append(MediaInfoTrack(track_type="Vídeo", attributes=tuple(video_attributes)))
            return tuple(tracks), None
        finally:
            capture.release()

    def _image_bit_depth(self, mode: str) -> str:
        mapping = {
            "1": "1 bit",
            "L": "8 bits",
            "P": "8 bits",
            "RGB": "24 bits",
            "RGBA": "32 bits",
            "CMYK": "32 bits",
            "I;16": "16 bits",
            "I": "32 bits",
            "F": "32 bits",
        }
        return mapping.get(mode, mode)

    def _format_file_size(self, size_bytes: int) -> str:
        units = ["bytes", "KB", "MB", "GB", "TB"]
        value = float(size_bytes)
        unit = units[0]
        for candidate in units:
            unit = candidate
            if value < 1024.0 or candidate == units[-1]:
                break
            value /= 1024.0
        if unit == "bytes":
            return f"{int(value)} bytes"
        return f"{value:.2f} {unit}"

    def _format_duration(self, total_seconds: float) -> str:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def _format_bitrate(self, size_bytes: int, duration_seconds: float) -> str | None:
        if duration_seconds <= 0.0:
            return None
        bits_per_second = (size_bytes * 8.0) / duration_seconds
        if bits_per_second >= 1_000_000:
            return f"{bits_per_second / 1_000_000:.2f} Mb/s"
        if bits_per_second >= 1_000:
            return f"{bits_per_second / 1_000:.2f} kb/s"
        return f"{bits_per_second:.0f} b/s"

    def _decode_fourcc(self, fourcc: int) -> str | None:
        if fourcc <= 0:
            return None
        decoded = "".join(chr((fourcc >> (8 * offset)) & 0xFF) for offset in range(4)).strip()
        normalized = "".join(character for character in decoded if character.isprintable()).strip()
        return normalized or None

    def _format_aspect_ratio(self, width: int, height: int) -> str:
        if width <= 0 or height <= 0:
            return "-"
        ratio = width / height
        return f"{ratio:.3f}:1"
