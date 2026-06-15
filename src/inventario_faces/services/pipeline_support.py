from __future__ import annotations

import json
import logging
import queue
import shutil
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import MediaType
from inventario_faces.domain.protocols import LogCallback, ProgressCallback
from inventario_faces.infrastructure.logging_setup import (
    format_exception_traceback,
    summarize_exception,
)
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.utils.path_utils import ensure_directory, file_io_path
from inventario_faces.utils.serialization import to_serializable


def emit_progress(
    progress_callback: ProgressCallback | None,
    current: int,
    total: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(current, total, message)


def emit_log(
    logger: logging.Logger,
    log_callback: LogCallback | None,
    message: str,
) -> None:
    logger.info(message)
    if log_callback is not None:
        log_callback(message)


def emit_exception(
    logger: logging.Logger,
    log_callback: LogCallback | None,
    context_message: str,
    exc: BaseException,
    *,
    include_traceback_in_callback: bool = False,
) -> tuple[str, str]:
    summary = summarize_exception(exc)
    traceback_text = format_exception_traceback(exc)
    logger.exception("%s | %s", context_message, summary)
    if log_callback is not None:
        log_callback(f"{context_message} | {summary}")
        if include_traceback_in_callback:
            log_callback("[Traceback] Inicio")
            log_callback(traceback_text)
            log_callback("[Traceback] Fim")
    return summary, traceback_text


def write_json_atomic(path: Path, payload: object) -> None:
    ensure_directory(path.parent)
    serialized = json.dumps(to_serializable(payload), indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(serialized)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def configuration_log_lines(config: AppConfig, providers: list[str]) -> list[str]:
    provider_label = ", ".join(providers) if providers else "selecao automatica"
    max_frames_label = (
        "sem limite"
        if config.video.max_frames_per_video is None
        else str(config.video.max_frames_per_video)
    )
    image_extensions = ", ".join(config.media.image_extensions)
    video_extensions = ", ".join(config.media.video_extensions)
    return [
        (
            "[Configuracao] Midias | "
            f"imagens={image_extensions} | videos={video_extensions}"
        ),
        (
            "[Configuracao] Video | "
            f"amostragem={config.video.sampling_interval_seconds:.2f}s | "
            f"max_quadros={max_frames_label} | "
            f"intervalo de keyframe={config.video.keyframe_interval_seconds:.2f}s | "
            f"mudanca significativa={config.video.significant_change_threshold:.2f}"
        ),
        (
            "[Configuracao] Tracking | "
            f"iou={config.tracking.iou_threshold:.2f} | "
            f"distancia={config.tracking.spatial_distance_threshold:.2f} | "
            f"embedding={config.tracking.embedding_similarity_threshold:.2f} | "
            f"score minimo={config.tracking.minimum_total_match_score:.2f} | "
            f"pesos=geo:{config.tracking.geometry_weight:.2f}/emb:{config.tracking.embedding_weight:.2f} | "
            f"perda maxima={config.tracking.max_missed_detections}"
        ),
        (
            "[Configuracao] Analise facial | "
            f"backend={config.face_model.backend} | "
            f"modelo={config.face_model.model_name} | "
            f"qualidade minima={config.face_model.minimum_face_quality:.2f} | "
            f"face minima={config.face_model.minimum_face_size_pixels}px | "
            f"provedores={provider_label}"
        ),
        (
            "[Configuracao] Clustering | "
            f"atribuicao={config.clustering.assignment_similarity:.2f} | "
            f"sugestao={config.clustering.candidate_similarity:.2f} | "
            f"grupo minimo={config.clustering.min_cluster_size} | "
            f"track minimo={config.clustering.min_track_size}"
        ),
        (
            "[Configuracao] Busca | "
            f"habilitada={'sim' if config.search.enabled else 'nao'} | "
            f"preferir_faiss={'sim' if config.search.prefer_faiss else 'nao'} | "
            f"coarse={config.search.coarse_top_k} | "
            f"refino={config.search.refine_top_k}"
        ),
        (
            "[Configuracao] Distribuicao | "
            f"habilitada={'sim' if config.distributed.enabled else 'nao'} | "
            f"execucao={config.distributed.execution_label} | "
            f"heartbeat={config.distributed.heartbeat_interval_seconds}s | "
            f"timeout_lock={config.distributed.stale_lock_timeout_minutes}min | "
            f"auto_finalizar={'sim' if config.distributed.auto_finalize else 'nao'} | "
            f"validar_parciais={'sim' if config.distributed.validate_partial_integrity else 'nao'} | "
            f"auto_recuperar={'sim' if config.distributed.auto_reprocess_invalid_partials else 'nao'}"
        ),
        (
            "[Configuracao] Aprimoramento | "
            f"pre_processamento={'sim' if config.enhancement.enable_preprocessing else 'nao'} | "
            f"brilho minimo={config.enhancement.minimum_brightness_to_enhance:.2f} | "
            f"gamma={config.enhancement.gamma:.2f} | "
            f"denoise={config.enhancement.denoise_strength}"
        ),
        (
            "[Configuracao] Relatorio | "
            f"pdf={'sim' if config.reporting.compile_pdf else 'nao'} | "
            f"max_tracks_por_grupo={config.reporting.max_tracks_per_group}"
        ),
        (
            "[Configuracao] Operacao | "
            f"copia_temporaria_local={'sim' if config.app.use_local_temp_copy else 'nao'}"
        ),
    ]


def copy_file_with_sha512(source_path: Path, target_path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    hasher = hashlib.sha512()
    ensure_directory(target_path.parent)
    with open(file_io_path(source_path), "rb") as source_stream, open(file_io_path(target_path), "wb") as target_stream:
        while True:
            chunk = source_stream.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            target_stream.write(chunk)
    return hasher.hexdigest()


def frames_with_original_source(
    frames: object,
    original_source_path: Path,
) -> object:
    return (
        replace(frame, source_path=original_source_path)
        for frame in frames
    )


#: Tamanho padrao do buffer de prefetch de quadros: equilibra sobreposicao decode/inferencia
#: e uso de memoria (quadros BGR mantidos simultaneamente).
DEFAULT_VIDEO_PREFETCH_BUFFER = 8


def prefetch_iterator(source: Iterable[Any], buffer_size: int = DEFAULT_VIDEO_PREFETCH_BUFFER) -> Iterator[Any]:
    """Consome ``source`` em uma thread de fundo, sobrepondo a decodificacao (CPU) com o
    trabalho do consumidor (ex.: inferencia em GPU).

    A ordem dos itens e preservada integralmente (produtor unico + fila FIFO), portanto os
    resultados sao identicos aos da iteracao sequencial -- a unica diferenca e temporal.
    Excecoes levantadas pelo produtor sao repropagadas ao consumidor na mesma posicao.
    """

    if buffer_size < 1:
        raise ValueError("buffer_size deve ser >= 1.")

    sentinel = object()
    item_queue: "queue.Queue[Any]" = queue.Queue(maxsize=buffer_size)
    error_box: list[BaseException] = []

    def _produce() -> None:
        try:
            for item in source:
                item_queue.put(item)
        except BaseException as exc:  # noqa: BLE001 - repropagado no consumidor
            error_box.append(exc)
        finally:
            item_queue.put(sentinel)

    worker = threading.Thread(target=_produce, name="video-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = item_queue.get()
            if item is sentinel:
                break
            yield item
        if error_box:
            raise error_box[0]
    finally:
        # Se o consumidor parar cedo (excecao ou break), drena a fila para liberar o
        # produtor que poderia estar bloqueado em put() numa fila cheia.
        try:
            while item_queue.get_nowait() is not sentinel:
                continue
        except queue.Empty:
            pass
        worker.join(timeout=1.0)


def cleanup_processing_input(cleanup_path: Path | None) -> None:
    if cleanup_path is None:
        return
    shutil.rmtree(cleanup_path, ignore_errors=True)


def prepare_processing_input(
    *,
    config: AppConfig,
    hashing_service: HashingService,
    file_path: Path,
    media_type: MediaType,
    file_prefix: str,
    text_logger: logging.Logger,
    log_callback: LogCallback | None,
) -> tuple[Path, str, Path | None]:
    if not config.app.use_local_temp_copy or media_type not in {MediaType.IMAGE, MediaType.VIDEO}:
        sha512 = hashing_service.sha512(file_path)
        emit_log(
            text_logger,
            log_callback,
            (
                f"{file_prefix} Entrada original mantida | "
                f"origem={file_path} | sha512={sha512[:16]}..."
            ),
        )
        return file_path, sha512, None

    temporary_directory = Path(tempfile.mkdtemp(prefix="inventario_faces_media_"))
    temporary_path = temporary_directory / file_path.name
    sha512 = copy_file_with_sha512(file_path, temporary_path)
    emit_log(
        text_logger,
        log_callback,
        (
            f"{file_prefix} Copia temporaria local preparada | "
            f"origem={file_path} | copia_local={temporary_path}"
        ),
    )
    return temporary_path, sha512, temporary_directory
