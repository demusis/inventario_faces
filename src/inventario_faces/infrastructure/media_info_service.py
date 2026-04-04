from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from inventario_faces.domain.entities import MediaInfoAttribute, MediaInfoTrack


class MediaInfoService:
    _EXECUTABLE_NAMES = (
        "mediainfo.exe",
        "MediaInfo.exe",
        "mediainfo",
        "MediaInfo",
    )

    _TRACK_LABELS = {
        "General": "Geral",
        "Image": "Imagem",
        "Video": "Vídeo",
        "Audio": "Áudio",
        "Text": "Texto",
    }

    _FIELD_LABELS = {
        "Format": "Formato",
        "Format_Profile": "Perfil do formato",
        "CodecID": "Identificador de codec",
        "InternetMediaType": "Tipo MIME",
        "Width": "Largura",
        "Height": "Altura",
        "DisplayAspectRatio/String": "Proporção de exibição",
        "Duration/String3": "Duração",
        "OverallBitRate/String": "Taxa de bits global",
        "BitRate/String": "Taxa de bits",
        "FrameRate/String": "Taxa de quadros",
        "FrameCount": "Número de quadros",
        "FileSize/String": "Tamanho do arquivo",
        "StreamSize/String": "Tamanho do fluxo",
        "ColorSpace": "Espaço de cor",
        "ChromaSubsampling": "Subamostragem de croma",
        "BitDepth": "Profundidade de bits",
        "Compression_Mode": "Modo de compressão",
        "ScanType": "Varredura",
        "ChannelLayout": "Layout de canais",
        "Channel_s_": "Canais",
        "Channels": "Canais",
        "SamplingRate/String": "Taxa de amostragem",
        "Language": "Idioma",
    }

    _TRACK_FIELDS = {
        "General": (
            "Format",
            "Format_Profile",
            "InternetMediaType",
            "FileSize/String",
            "Duration/String3",
            "OverallBitRate/String",
            "FrameCount",
        ),
        "Image": (
            "Format",
            "InternetMediaType",
            "Width",
            "Height",
            "ColorSpace",
            "ChromaSubsampling",
            "BitDepth",
            "Compression_Mode",
            "StreamSize/String",
        ),
        "Video": (
            "Format",
            "Format_Profile",
            "CodecID",
            "Width",
            "Height",
            "DisplayAspectRatio/String",
            "Duration/String3",
            "BitRate/String",
            "FrameRate/String",
            "FrameCount",
            "ScanType",
            "ColorSpace",
            "ChromaSubsampling",
            "BitDepth",
            "StreamSize/String",
        ),
        "Audio": (
            "Format",
            "CodecID",
            "Duration/String3",
            "BitRate/String",
            "Channels",
            "ChannelLayout",
            "SamplingRate/String",
            "BitDepth",
            "Language",
            "StreamSize/String",
        ),
        "Text": (
            "Format",
            "CodecID",
            "Language",
            "StreamSize/String",
        ),
    }

    def __init__(
        self,
        executable_path: str | None = None,
        directory: str | Path | None = None,
    ) -> None:
        self._configured_directory = (
            Path(directory).expanduser().resolve()
            if directory not in (None, "")
            else None
        )
        self._executable_path = self._resolve_executable_path(executable_path, self._configured_directory)

    @property
    def executable_path(self) -> str | None:
        return self._executable_path

    def extract(self, path: Path) -> tuple[tuple[MediaInfoTrack, ...], str | None]:
        if not self._executable_path:
            if self._configured_directory is not None:
                return (), (
                    "MediaInfo nao encontrado no diretorio configurado: "
                    f"{self._configured_directory}"
                )
            return (), "MediaInfo nao disponivel no ambiente."

        try:
            completed = subprocess.run(
                [self._executable_path, "--Output=JSON", str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
        except Exception as exc:
            return (), f"Falha ao executar MediaInfo: {exc}"

        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "MediaInfo retornou erro."
            return (), message

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            return (), f"Falha ao interpretar a saida JSON do MediaInfo: {exc}"

        tracks_payload = payload.get("media", {}).get("track", [])
        tracks: list[MediaInfoTrack] = []
        for track in tracks_payload:
            if not isinstance(track, dict):
                continue
            track_type = str(track.get("@type", "")).strip()
            if not track_type:
                continue
            attributes = self._extract_attributes(track_type, track)
            if not attributes:
                continue
            tracks.append(
                MediaInfoTrack(
                    track_type=self._TRACK_LABELS.get(track_type, track_type),
                    attributes=tuple(attributes),
                )
            )
        return tuple(tracks), None

    def _resolve_executable_path(
        self,
        executable_path: str | None,
        configured_directory: Path | None,
    ) -> str | None:
        if executable_path:
            return str(Path(executable_path).expanduser().resolve())

        configured_candidate = self._resolve_from_directory(configured_directory)
        if configured_candidate is not None:
            return configured_candidate
        return shutil.which("mediainfo")

    def _resolve_from_directory(self, configured_directory: Path | None) -> str | None:
        if configured_directory is None:
            return None
        if configured_directory.is_file():
            return str(configured_directory)
        if not configured_directory.exists():
            return None
        for executable_name in self._EXECUTABLE_NAMES:
            candidate = configured_directory / executable_name
            if candidate.exists():
                return str(candidate.resolve())
        return None

    def _extract_attributes(self, track_type: str, payload: dict[str, object]) -> list[MediaInfoAttribute]:
        fields = self._TRACK_FIELDS.get(track_type, ())
        attributes: list[MediaInfoAttribute] = []
        for field_name in fields:
            raw_value = payload.get(field_name)
            if raw_value in (None, ""):
                continue
            text = str(raw_value).strip()
            if not text:
                continue
            attributes.append(
                MediaInfoAttribute(
                    label=self._FIELD_LABELS.get(field_name, field_name),
                    value=text,
                )
            )
        return attributes
