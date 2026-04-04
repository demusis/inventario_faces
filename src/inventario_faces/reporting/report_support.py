from __future__ import annotations

from inventario_faces import __version__
from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import FaceCluster, KeyFrame, SearchArtifacts
from inventario_faces.utils.latex import format_seconds

PROJECT_URL = "https://github.com/demusis/inventario_faces"

_KEYFRAME_REASON_LABELS = {
    "track_start": "início do track",
    "initial_reference": "primeira referência do track",
    "interval": "intervalo temporal",
    "significant_change": "mudança significativa",
    "quality_peak": "pico de qualidade",
}


def candidate_cluster_map(clusters: list[FaceCluster]) -> dict[str, list[str]]:
    return {
        cluster.cluster_id: sorted(cluster.candidate_cluster_ids)
        for cluster in clusters
        if cluster.candidate_cluster_ids
    }


def keyframe_reason_labels(keyframe: KeyFrame) -> list[str]:
    return [
        _KEYFRAME_REASON_LABELS.get(reason, reason.replace("_", " "))
        for reason in keyframe.selection_reasons
    ]


def keyframe_reference_text(keyframe: KeyFrame) -> str:
    parts = ["Quadro de referência do track"]
    if keyframe.frame_index is not None:
        parts[0] = f"Quadro de referência do track: {keyframe.frame_index:06d}"
    parts.append(f"instante {format_seconds(keyframe.timestamp_seconds)}")
    reasons = keyframe_reason_labels(keyframe)
    if reasons:
        parts.append(f"critérios de seleção: {', '.join(reasons)}")
    return "; ".join(parts) + "."


def media_track_type_label(track_type: str) -> str:
    normalized = track_type.strip().lower()
    if normalized in {"geral", "general"}:
        return "Arquivo"
    return track_type


def format_group_similarity(mean_similarity: float | None, track_count: int) -> str:
    if track_count < 2:
        return "n/a (grupo unitário)"
    if mean_similarity is None:
        return "n/a (embeddings indisponíveis)"
    return f"{mean_similarity:.3f}"


def software_reference_abnt_text() -> str:
    return (
        "DEMUSIS. Inventario Faces. [S. l.], 2026. "
        f"Disponível em: {PROJECT_URL}. Acesso em: 4 abr. 2026."
    )


def software_reference_abnt_latex() -> str:
    return (
        r"DEMUSIS. \textit{Inventario Faces}. [S.\ l.], 2026. "
        rf"Dispon\'ivel em: \url{{{PROJECT_URL}}}. Acesso em: 4 abr. 2026."
    )


def inventory_methodology_items(config: AppConfig, search: SearchArtifacts | None) -> list[str]:
    search_engine = search.engine if search is not None else "desabilitado"
    max_frames_text = (
        "sem limite explícito de quadros amostrados por vídeo"
        if config.video.max_frames_per_video is None
        else f"limitado a {config.video.max_frames_per_video} quadros amostrados por vídeo"
    )
    return [
        "Varredura recursiva do diretório de entrada, com cálculo de hash SHA-512 para cada arquivo e extração interna de metadados técnicos de imagem e vídeo.",
        (
            "Quando o modo distribuído está habilitado, o lote passa a operar com manifesto global, locks por arquivo e retomada entre instâncias, "
            "permitindo divisão segura do processamento entre múltiplos computadores ou execuções concorrentes. "
            "Antes da consolidação final, os parciais podem ser validados quanto à integridade e, quando configurado, os itens ausentes ou corrompidos são reprocessados automaticamente."
            if config.distributed.enabled
            else "A coordenação do lote ocorre localmente nesta instância, sem mecanismo de divisão de trabalho entre múltiplos computadores."
        ),
        (
            "Para vídeos, a análise facial não é exaustivamente quadro a quadro por padrão: "
            f"o sistema extrai quadros em intervalos temporais configuráveis e, nesta configuração, "
            f"processa aproximadamente um quadro a cada {config.video.sampling_interval_seconds:.2f} segundos, "
            f"{max_frames_text}, sem alteração dos arquivos originais."
        ),
        (
            "A detecção facial é aplicada em cada quadro amostrado, e os tracks são formados pela associação temporal "
            "das detecções consecutivas, combinando continuidade geométrica e, quando necessário, similaridade de embedding "
            "para desambiguação."
        ),
        (
            "Aplicação controlada de pré-processamento apenas quando configurado e necessário, com registro auditável das melhorias "
            "executadas sobre os derivados e sem sobrescrever a evidência original."
        ),
        (
            "Seleção de keyframes representativos por início de track, intervalo temporal, mudança significativa e ganho de qualidade, "
            "considerando detecção, nitidez, frontalidade e iluminação."
        ),
        (
            "Representação de cada track por embedding médio normalizado, estatísticas de qualidade e melhores recortes associados."
        ),
        (
            f"Agrupamento em nível de track, com limiar de atribuição {config.clustering.assignment_similarity:.2f} "
            f"e limiar de sugestão intergrupos {config.clustering.candidate_similarity:.2f}."
        ),
        (
            f"Indexação vetorial para busca em duas etapas, com mecanismo atual {search_engine}: "
            "busca coarse em grupos e tracks, seguida de refinamento em ocorrências internas."
        ),
    ]


def face_search_methodology_items(selected_track_id: str) -> list[str]:
    return [
        "O diretório-alvo é processado pelo mesmo pipeline orientado a tracks usado no inventário principal, preservando hashes, logs e artefatos derivativos auditáveis.",
        "A imagem de consulta passa pelo mesmo fluxo de detecção e filtragem, mantendo apenas faces elegíveis segundo os limiares configurados.",
        (
            f"Quando há mais de uma face elegível na consulta, o sistema seleciona automaticamente a face de melhor qualidade, "
            f"registrada como {selected_track_id}, para servir como referência da busca."
        ),
        "A pesquisa vetorial ocorre em duas etapas: recuperação coarse de grupos e tracks candidatos, seguida de refinamento nas ocorrências internas compatíveis.",
    ]


def technical_parameter_items(
    config: AppConfig,
    search: SearchArtifacts | None,
) -> list[str]:
    search_engine = search.engine if search is not None else "desabilitado"
    search_vectors = (
        f"tracks={search.track_vector_count}; grupos={search.cluster_vector_count}"
        if search is not None
        else "tracks=0; grupos=0"
    )
    det_size = (
        f"{config.face_model.det_size[0]}x{config.face_model.det_size[1]}"
        if config.face_model.det_size is not None
        else "resolução original do quadro"
    )
    providers = ", ".join(config.face_model.providers) if config.face_model.providers else "seleção automática"
    max_frames = (
        "sem limite explícito"
        if config.video.max_frames_per_video is None
        else str(config.video.max_frames_per_video)
    )
    image_extensions = ", ".join(config.media.image_extensions)
    video_extensions = ", ".join(config.media.video_extensions)

    return [
        (
            f"Aplicativo: nome={config.app.name}; versão={__version__}; saída={config.app.output_directory_name}; "
            f"título={config.app.report_title}; organização={config.app.organization}; log={config.app.log_level}; "
            f"cópia temporária local={'sim' if config.app.use_local_temp_copy else 'não'}."
        ),
        (
            f"Mídias: imagens={image_extensions}; vídeos={video_extensions}."
        ),
        (
            f"Vídeo: amostragem={config.video.sampling_interval_seconds:.2f}s; máximo de quadros={max_frames}; "
            f"intervalo mínimo entre keyframes={config.video.keyframe_interval_seconds:.2f}s; "
            f"mudança significativa={config.video.significant_change_threshold:.2f}."
        ),
        (
            f"Modelo facial: backend={config.face_model.backend}; modelo={config.face_model.model_name}; "
            f"det_size={det_size}; qualidade mínima={config.face_model.minimum_face_quality:.2f}; "
            f"tamanho mínimo={config.face_model.minimum_face_size_pixels}px; ctx_id={config.face_model.ctx_id}; "
            f"providers={providers}."
        ),
        (
            f"Tracking: IoU={config.tracking.iou_threshold:.2f}; distância espacial={config.tracking.spatial_distance_threshold:.2f}; "
            f"similaridade de embedding={config.tracking.embedding_similarity_threshold:.2f}; "
            f"pontuação total mínima={config.tracking.minimum_total_match_score:.2f}; "
            f"peso geométrico={config.tracking.geometry_weight:.2f}; peso do embedding={config.tracking.embedding_weight:.2f}; "
            f"perdas máximas={config.tracking.max_missed_detections}; margem de confiança={config.tracking.confidence_margin:.2f}; "
            f"embeddings representativos={config.tracking.representative_embeddings_per_track}; "
            f"melhores recortes={config.tracking.top_crops_per_track}; "
            f"ganho mínimo de qualidade={config.tracking.quality_improvement_margin:.2f}."
        ),
        (
            f"Clustering: atribuição={config.clustering.assignment_similarity:.2f}; "
            f"sugestão entre grupos={config.clustering.candidate_similarity:.2f}; "
            f"grupo mínimo={config.clustering.min_cluster_size}; track mínimo={config.clustering.min_track_size}."
        ),
        (
            f"Busca vetorial: habilitada={'sim' if config.search.enabled else 'não'}; "
            f"preferir FAISS={'sim' if config.search.prefer_faiss else 'não'}; "
            f"coarse_top_k={config.search.coarse_top_k}; refine_top_k={config.search.refine_top_k}; "
            f"engine={search_engine}; vetores indexados={search_vectors}."
        ),
        (
            f"Distribuição: habilitada={'sim' if config.distributed.enabled else 'não'}; "
            f"execução={config.distributed.execution_label}; nó={config.distributed.node_name or 'hostname automático'}; "
            f"heartbeat={config.distributed.heartbeat_interval_seconds}s; "
            f"timeout de lock={config.distributed.stale_lock_timeout_minutes}min; "
            f"auto-finalização={'sim' if config.distributed.auto_finalize else 'não'}; "
            f"validar parciais={'sim' if config.distributed.validate_partial_integrity else 'não'}; "
            f"auto-recuperar parciais={'sim' if config.distributed.auto_reprocess_invalid_partials else 'não'}."
        ),
        (
            f"Aprimoramento: pré-processamento={'sim' if config.enhancement.enable_preprocessing else 'não'}; "
            f"brilho mínimo={config.enhancement.minimum_brightness_to_enhance:.2f}; "
            f"CLAHE clip={config.enhancement.clahe_clip_limit:.2f}; "
            f"CLAHE tile={config.enhancement.clahe_tile_grid_size}; gamma={config.enhancement.gamma:.2f}; "
            f"denoise={config.enhancement.denoise_strength}."
        ),
        (
            f"Relatório: compilar PDF={'sim' if config.reporting.compile_pdf else 'não'}; "
            f"máximo de tracks por grupo={config.reporting.max_tracks_per_group}."
        ),
        (
            f"Nota institucional de cadeia de custódia: {config.forensics.chain_of_custody_note}"
        ),
    ]
