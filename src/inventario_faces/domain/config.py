from __future__ import annotations

from dataclasses import dataclass, field


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} nao pode ficar em branco.")
    return cleaned


def _validate_probability(value: float, field_name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} deve ficar entre 0.0 e 1.0.")


def _validate_positive_number(value: float, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} deve ser maior que zero.")


def _validate_non_negative_number(value: float, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} nao pode ser negativo.")


def _normalize_extensions(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_value in values:
        text = str(raw_value).strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = f".{text}"
        if text == ".":
            raise ValueError(f"{field_name} contem uma extensao invalida.")
        if text not in normalized:
            normalized.append(text)
    if not normalized:
        raise ValueError(f"{field_name} precisa informar ao menos uma extensao.")
    return tuple(normalized)


def _normalize_string_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_value in values:
        text = str(raw_value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


@dataclass(frozen=True)
class AppSettings:
    """Configuracoes institucionais e operacionais globais da aplicacao."""

    name: str
    output_directory_name: str
    report_title: str
    organization: str
    log_level: str = "INFO"
    use_local_temp_copy: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_non_empty_text(self.name, "Nome do aplicativo"))
        output_directory_name = _normalize_non_empty_text(
            self.output_directory_name,
            "Diretorio de saida",
        )
        if output_directory_name in {".", ".."} or any(
            separator in output_directory_name for separator in ("/", "\\", ":")
        ):
            raise ValueError(
                "Diretorio de saida deve ser apenas o nome da pasta derivada, sem caminho absoluto ou relativo."
            )
        object.__setattr__(self, "output_directory_name", output_directory_name)
        object.__setattr__(
            self,
            "report_title",
            _normalize_non_empty_text(self.report_title, "Titulo do relatorio"),
        )
        object.__setattr__(
            self,
            "organization",
            _normalize_non_empty_text(self.organization, "Organizacao responsavel"),
        )
        log_level = _normalize_non_empty_text(self.log_level, "Nivel de log").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("Nivel de log deve ser DEBUG, INFO, WARNING ou ERROR.")
        object.__setattr__(self, "log_level", log_level)


@dataclass(frozen=True)
class MediaSettings:
    """Extensoes de arquivos elegiveis para processamento."""

    image_extensions: tuple[str, ...]
    video_extensions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_extensions",
            _normalize_extensions(self.image_extensions, "Extensoes de imagem"),
        )
        object.__setattr__(
            self,
            "video_extensions",
            _normalize_extensions(self.video_extensions, "Extensoes de video"),
        )


@dataclass(frozen=True)
class VideoSettings:
    """Parametros de amostragem temporal e reducao de redundancia em videos."""

    sampling_interval_seconds: float
    max_frames_per_video: int | None = None
    keyframe_interval_seconds: float = 3.0
    significant_change_threshold: float = 0.18

    def __post_init__(self) -> None:
        _validate_positive_number(self.sampling_interval_seconds, "Intervalo de amostragem")
        if self.max_frames_per_video is not None and self.max_frames_per_video <= 0:
            raise ValueError("Maximo de quadros por video deve ser maior que zero quando informado.")
        _validate_positive_number(self.keyframe_interval_seconds, "Intervalo minimo entre keyframes")
        _validate_probability(self.significant_change_threshold, "Limiar de mudanca significativa")


@dataclass(frozen=True)
class FaceModelSettings:
    """Configuracao do backend de deteccao e extracao de embeddings faciais."""

    backend: str
    model_name: str
    det_size: tuple[int, int] | None
    minimum_face_quality: float = 0.6
    minimum_face_size_pixels: int = 40
    ctx_id: int = 0
    providers: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _normalize_non_empty_text(self.backend, "Backend facial"))
        object.__setattr__(self, "model_name", _normalize_non_empty_text(self.model_name, "Modelo facial"))
        if self.det_size is not None:
            if len(self.det_size) != 2 or any(int(value) <= 0 for value in self.det_size):
                raise ValueError("det_size deve conter largura e altura positivas.")
            object.__setattr__(self, "det_size", (int(self.det_size[0]), int(self.det_size[1])))
        _validate_probability(self.minimum_face_quality, "Qualidade minima da face")
        if self.minimum_face_size_pixels <= 0:
            raise ValueError("Tamanho minimo da face deve ser maior que zero.")
        object.__setattr__(self, "providers", _normalize_string_tuple(self.providers))


@dataclass(frozen=True)
class ClusteringSettings:
    """Limiarizacao e filtros do agrupamento probabilistico por track."""

    assignment_similarity: float
    candidate_similarity: float
    min_cluster_size: int = 1
    min_track_size: int = 1

    def __post_init__(self) -> None:
        _validate_probability(self.assignment_similarity, "Similaridade de atribuicao")
        _validate_probability(self.candidate_similarity, "Similaridade candidata")
        if self.candidate_similarity > self.assignment_similarity:
            raise ValueError(
                "Similaridade candidata nao pode ser maior que a similaridade de atribuicao."
            )
        if self.min_cluster_size <= 0:
            raise ValueError("Tamanho minimo do grupo deve ser maior que zero.")
        if self.min_track_size <= 0:
            raise ValueError("Tamanho minimo do track deve ser maior que zero.")


@dataclass(frozen=True)
class TrackingSettings:
    """Parametros de associacao temporal das deteccoes faciais."""

    iou_threshold: float = 0.15
    spatial_distance_threshold: float = 0.18
    embedding_similarity_threshold: float = 0.48
    minimum_total_match_score: float = 0.30
    geometry_weight: float = 0.45
    embedding_weight: float = 0.55
    max_missed_detections: int = 2
    confidence_margin: float = 0.05
    representative_embeddings_per_track: int = 5
    top_crops_per_track: int = 4
    quality_improvement_margin: float = 0.05

    def __post_init__(self) -> None:
        _validate_probability(self.iou_threshold, "IoU do tracking")
        _validate_probability(self.spatial_distance_threshold, "Distancia espacial do tracking")
        _validate_probability(self.embedding_similarity_threshold, "Similaridade de embedding do tracking")
        _validate_probability(self.minimum_total_match_score, "Pontuacao minima total do tracking")
        _validate_non_negative_number(self.geometry_weight, "Peso geometrico")
        _validate_non_negative_number(self.embedding_weight, "Peso do embedding")
        if self.geometry_weight == 0 and self.embedding_weight == 0:
            raise ValueError("Ao menos um dos pesos do tracking deve ser maior que zero.")
        if self.max_missed_detections < 0:
            raise ValueError("Maximo de perdas do tracking nao pode ser negativo.")
        _validate_probability(self.confidence_margin, "Margem de confianca do tracking")
        if self.representative_embeddings_per_track <= 0:
            raise ValueError("Embeddings representativos por track deve ser maior que zero.")
        if self.top_crops_per_track <= 0:
            raise ValueError("Melhores recortes por track deve ser maior que zero.")
        _validate_probability(self.quality_improvement_margin, "Margem de melhoria de qualidade")


@dataclass(frozen=True)
class EnhancementSettings:
    """Parametros do pre-processamento auditavel aplicado aos derivados."""

    enable_preprocessing: bool = True
    minimum_brightness_to_enhance: float = 0.36
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    gamma: float = 1.0
    denoise_strength: int = 0

    def __post_init__(self) -> None:
        _validate_probability(self.minimum_brightness_to_enhance, "Brilho minimo para aprimoramento")
        _validate_positive_number(self.clahe_clip_limit, "CLAHE clip limit")
        if self.clahe_tile_grid_size <= 0:
            raise ValueError("CLAHE tile grid size deve ser maior que zero.")
        _validate_positive_number(self.gamma, "Gamma")
        if self.denoise_strength < 0:
            raise ValueError("Denoise nao pode ser negativo.")


@dataclass(frozen=True)
class SearchSettings:
    """Parametros da indexacao vetorial e da busca coarse-to-fine."""

    enabled: bool = True
    prefer_faiss: bool = True
    coarse_top_k: int = 8
    refine_top_k: int = 12

    def __post_init__(self) -> None:
        if self.coarse_top_k <= 0:
            raise ValueError("Coarse top-k deve ser maior que zero.")
        if self.refine_top_k <= 0:
            raise ValueError("Refine top-k deve ser maior que zero.")


@dataclass(frozen=True)
class LikelihoodRatioSettings:
    """Parametros da calibracao LR baseada em KDE para comparacao entre conjuntos."""

    max_scores_per_distribution: int = 20000
    minimum_identities_with_faces: int = 2
    minimum_same_source_scores: int = 5
    minimum_different_source_scores: int = 5
    minimum_unique_scores_per_distribution: int = 2
    kde_bandwidth_scale: float = 1.0
    kde_uniform_floor_weight: float = 0.001
    kde_min_density: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_scores_per_distribution <= 0:
            raise ValueError("Maximo de scores por distribuicao deve ser maior que zero.")
        if self.minimum_identities_with_faces < 2:
            raise ValueError("Minimo de identidades com faces deve ser ao menos 2.")
        if self.minimum_same_source_scores <= 0:
            raise ValueError("Minimo de scores de mesma origem deve ser maior que zero.")
        if self.minimum_different_source_scores <= 0:
            raise ValueError("Minimo de scores de origem distinta deve ser maior que zero.")
        if self.minimum_unique_scores_per_distribution < 2:
            raise ValueError("Minimo de scores unicos por distribuicao deve ser ao menos 2.")
        _validate_positive_number(self.kde_bandwidth_scale, "Escala de banda da KDE")
        _validate_probability(self.kde_uniform_floor_weight, "Peso do piso uniforme da KDE")
        if self.kde_uniform_floor_weight >= 1.0:
            raise ValueError("Peso do piso uniforme da KDE deve ser menor que 1.0.")
        _validate_positive_number(self.kde_min_density, "Densidade minima da KDE")


@dataclass(frozen=True)
class DistributedSettings:
    """Coordenacao de lotes compartilhados entre nos ou instancias."""

    enabled: bool = False
    execution_label: str = "compartilhado"
    node_name: str | None = None
    heartbeat_interval_seconds: int = 15
    stale_lock_timeout_minutes: int = 120
    auto_finalize: bool = True
    validate_partial_integrity: bool = True
    auto_reprocess_invalid_partials: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_label",
            _normalize_non_empty_text(self.execution_label, "Identificador da execucao compartilhada"),
        )
        if self.node_name is not None:
            normalized_node_name = self.node_name.strip()
            object.__setattr__(self, "node_name", normalized_node_name or None)
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("Heartbeat do no deve ser maior que zero.")
        if self.stale_lock_timeout_minutes <= 0:
            raise ValueError("Tempo para lock orfao deve ser maior que zero.")
        if self.auto_reprocess_invalid_partials and not self.validate_partial_integrity:
            raise ValueError(
                "Recuperacao automatica de parciais exige validacao de integridade habilitada."
            )


@dataclass(frozen=True)
class ReportingSettings:
    """Controles de geracao e densidade dos relatorios tecnicos."""

    compile_pdf: bool = True
    max_tracks_per_group: int = 8

    def __post_init__(self) -> None:
        if self.max_tracks_per_group <= 0:
            raise ValueError("Maximo de tracks por grupo deve ser maior que zero.")


@dataclass(frozen=True)
class ForensicsSettings:
    """Notas institucionais e premissas de cadeia de custodia."""

    chain_of_custody_note: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chain_of_custody_note",
            _normalize_non_empty_text(self.chain_of_custody_note, "Nota de cadeia de custodia"),
        )


@dataclass(frozen=True)
class AppConfig:
    """Configuracao consolidada do aplicativo e de todo o pipeline."""

    app: AppSettings
    media: MediaSettings
    video: VideoSettings
    face_model: FaceModelSettings
    clustering: ClusteringSettings
    reporting: ReportingSettings
    forensics: ForensicsSettings
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    enhancement: EnhancementSettings = field(default_factory=EnhancementSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    likelihood_ratio: LikelihoodRatioSettings = field(default_factory=LikelihoodRatioSettings)
    distributed: DistributedSettings = field(default_factory=DistributedSettings)
