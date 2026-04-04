from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from inventario_faces.domain.config import (
    AppConfig,
    AppSettings,
    ClusteringSettings,
    EnhancementSettings,
    FaceModelSettings,
    ForensicsSettings,
    MediaSettings,
    ReportingSettings,
    SearchSettings,
    TrackingSettings,
    VideoSettings,
)

INSIGHTFACE_URL = "https://github.com/deepinsight/insightface"
ARCFACE_URL = (
    "https://openaccess.thecvf.com/content_CVPR_2019/html/"
    "Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html"
)
SCRFD_URL = "https://arxiv.org/abs/2105.04714"
ONNXRUNTIME_EP_URL = "https://onnxruntime.ai/docs/execution-providers/"
OPENCV_VIDEOCAPTURE_URL = "https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html"
MEDIAINFO_URL = "https://mediaarea.net/en/MediaInfo"
FAISS_URL = "https://github.com/facebookresearch/faiss"
ABNT_ACCESS_DATE = "Acesso em: 4 abr. 2026."


class ConfigDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_config = config
        self.setWindowTitle("Configurações do procedimento")
        self.resize(980, 860)
        self._build_ui()
        self._load_config(config)

    @property
    def selected_config(self) -> AppConfig:
        return self._selected_config

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        description = QLabel(
            (
                "Revise os parâmetros do pipeline antes de cada execução. "
                "As abas organizam amostragem, tracking, clustering, busca, aprimoramento e relatório. "
                "Os metadados técnicos da mídia agora são extraídos internamente, sem depender do aplicativo externo MediaInfo."
            )
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        tabs = QTabWidget(self)
        tabs.addTab(self._build_general_tab(), "Geral")
        tabs.addTab(self._build_media_tab(), "Mídias")
        tabs.addTab(self._build_face_tab(), "Análise facial")
        tabs.addTab(self._build_tracking_tab(), "Tracking")
        tabs.addTab(self._build_clustering_search_tab(), "Agrupamento e busca")
        tabs.addTab(self._build_enhancement_tab(), "Aprimoramento")
        tabs.addTab(self._build_reporting_tab(), "Relatório")
        layout.addWidget(tabs)

        self._help_title_label = QLabel("Ajuda técnica")
        self._help_title_label.setStyleSheet("font-weight: bold;")
        self._help_body = QTextBrowser(self)
        self._help_body.setReadOnly(True)
        self._help_body.setOpenExternalLinks(True)
        self._help_body.setMinimumHeight(220)
        self._help_body.setHtml(
            self._help_html(
                definition=(
                    "Clique no botão ? ao lado de um campo para abrir a descrição técnica, "
                    "o impacto operacional e recomendações práticas de calibração."
                ),
                operational_effect=(
                    "As alterações persistem para as próximas execuções. "
                    "Evite mudar muitos parâmetros ao mesmo tempo sem registrar a justificativa."
                ),
                recommendation=(
                    "Em produção, ajuste limiares com material representativo do caso e mantenha "
                    "coerência entre o relatório técnico e a configuração realmente usada."
                ),
                references=[
                    ("InsightFace", INSIGHTFACE_URL),
                    ("OpenCV VideoCapture", OPENCV_VIDEOCAPTURE_URL),
                    ("ONNX Runtime execution providers", ONNXRUNTIME_EP_URL),
                ],
            )
        )
        layout.addWidget(self._help_title_label)
        layout.addWidget(self._help_body)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        apply_button = button_box.button(QDialogButtonBox.Ok)
        cancel_button = button_box.button(QDialogButtonBox.Cancel)
        if apply_button is not None:
            apply_button.setText("Aplicar")
        if cancel_button is not None:
            cancel_button.setText("Cancelar")

        restore_button = QPushButton("Restaurar valores carregados", self)
        restore_button.clicked.connect(self._restore_selected_config)
        button_box.addButton(restore_button, QDialogButtonBox.ResetRole)
        button_box.accepted.connect(self._accept_configuration)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _build_general_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._output_directory_input = QLineEdit(tab)
        self._report_title_input = QLineEdit(tab)
        self._organization_input = QLineEdit(tab)
        self._log_level_combo = QComboBox(tab)
        self._log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])

        form.addRow("Diretório de saída:", self._with_help(self._output_directory_input, "Diretório de saída", self._help_html(
            definition="Nome da pasta derivada criada dentro do diretório examinado para armazenar artefatos, inventários, logs e relatórios.",
            operational_effect="Mantém o material derivado segregado dos arquivos originais e facilita rastreabilidade forense.",
            recommendation="Use um nome estável e institucional, como inventario_faces_output.",
        )))
        form.addRow("Título do relatório:", self._with_help(self._report_title_input, "Título do relatório", self._help_html(
            definition="Título formal exibido na capa e nas seções do relatório técnico.",
            operational_effect="Ajuda a identificar o escopo do documento e a diferenciar versões ou procedimentos.",
            recommendation="Prefira formulação descritiva e neutra, sem sugerir identificação conclusiva.",
            references=[("ArcFace", ARCFACE_URL)],
        )))
        form.addRow("Organização responsável:", self._with_help(self._organization_input, "Organização responsável", self._help_html(
            definition="Unidade, laboratório, delegacia ou setor responsável pela emissão do relatório.",
            operational_effect="Integra a cadeia documental do exame e facilita rastreio institucional.",
            recommendation="Use a denominação oficial completa da unidade emissora.",
        )))
        form.addRow("Nível de log:", self._with_help(self._log_level_combo, "Nível de log", self._help_html(
            definition="Controla a verbosidade dos registros textuais e estruturados da execução.",
            operational_effect="Níveis mais altos ampliam auditabilidade e diagnóstico, mas aumentam o volume persistido.",
            recommendation="Use INFO na rotina. Use DEBUG para calibração, homologação e investigação de falhas.",
        )))
        return tab

    def _build_media_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._image_extensions_input = QLineEdit(tab)
        self._image_extensions_input.setPlaceholderText(".jpg, .jpeg, .png")
        self._video_extensions_input = QLineEdit(tab)
        self._video_extensions_input.setPlaceholderText(".mp4, .avi, .mkv, .dav")

        self._sampling_interval_spin = self._double_spin_box(0.1, 3600.0, 2, 0.1, " s")
        self._keyframe_interval_spin = self._double_spin_box(0.1, 3600.0, 2, 0.1, " s")
        self._significant_change_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)

        self._max_frames_spin = QSpinBox(tab)
        self._max_frames_spin.setRange(1, 1_000_000)
        self._unlimited_frames_checkbox = QCheckBox("Sem limite de quadros por vídeo", tab)
        self._unlimited_frames_checkbox.toggled.connect(self._max_frames_spin.setDisabled)

        max_frames_widget = QWidget(tab)
        max_frames_layout = QVBoxLayout(max_frames_widget)
        max_frames_layout.setContentsMargins(0, 0, 0, 0)
        max_frames_layout.addWidget(self._max_frames_spin)
        max_frames_layout.addWidget(self._unlimited_frames_checkbox)

        form.addRow("Extensões de imagem:", self._with_help(self._image_extensions_input, "Extensões de imagem", self._help_html(
            definition="Lista de extensões tratadas como imagem estática pelo scanner.",
            operational_effect="Controla quais arquivos entram no pipeline de leitura de imagem.",
            recommendation="Inclua apenas formatos realmente presentes e suportados no ambiente.",
        )))
        form.addRow("Extensões de vídeo:", self._with_help(self._video_extensions_input, "Extensões de vídeo", self._help_html(
            definition="Lista de extensões consideradas vídeo para abertura, amostragem e extração de quadros.",
            operational_effect="Se estiver ampla demais, o sistema tentará abrir arquivos indevidos; se restrita demais, mídias pertinentes podem ser omitidas. Arquivos .dav podem ser tratados como vídeo quando a estação tiver decoder compatível.",
            recommendation="Mantenha apenas formatos homologados na estação, incluindo .dav quando esse tipo de arquivo fizer parte do acervo.",
            references=[("OpenCV VideoCapture", OPENCV_VIDEOCAPTURE_URL)],
        )))
        form.addRow("Intervalo de amostragem:", self._with_help(self._sampling_interval_spin, "Intervalo de amostragem", self._help_html(
            definition="Espaçamento temporal entre quadros extraídos de cada vídeo para análise facial. Por padrão, o sistema não processa todos os quadros originais do arquivo.",
            operational_effect="Intervalos menores aumentam cobertura temporal e custo computacional; intervalos maiores reduzem custo, mas podem perder aparições breves.",
            recommendation="Comece em 1,0 s a 2,0 s e reduza apenas quando o caso exigir maior granularidade temporal. Para se aproximar de uma análise quadro a quadro, o intervalo precisa cair até equivaler a passo de 1 frame no FPS do vídeo.",
            references=[("OpenCV VideoCapture", OPENCV_VIDEOCAPTURE_URL)],
        )))
        form.addRow("Máximo de quadros por vídeo:", self._with_help(max_frames_widget, "Máximo de quadros por vídeo", self._help_html(
            definition="Limite superior de quadros amostrados por arquivo. Quando desativado, o vídeo é percorrido conforme o intervalo de amostragem.",
            operational_effect="Controla custo de processamento, tamanho dos artefatos e duração da execução em vídeos longos.",
            recommendation="Use limite em triagens exploratórias; desative apenas quando houver justificativa para cobertura integral.",
        )))
        form.addRow("Intervalo mínimo entre keyframes:", self._with_help(self._keyframe_interval_spin, "Intervalo mínimo entre keyframes", self._help_html(
            definition="Tempo mínimo entre keyframes consecutivos do mesmo track, salvo quando houver início de track ou mudança significativa.",
            operational_effect="Reduz redundância de frames e limita embeddings calculados em sequência muito curta.",
            recommendation="Use valor compatível com a dinâmica do vídeo; 3 a 5 segundos é um ponto inicial útil.",
        )))
        form.addRow("Limiar de mudança significativa:", self._with_help(self._significant_change_spin, "Limiar de mudança significativa", self._help_html(
            definition="Pontuação mínima de mudança visual ou geométrica para justificar novo keyframe antes de vencer o intervalo temporal.",
            operational_effect="Valores menores geram mais keyframes; valores maiores reduzem redundância, mas podem perder variações relevantes.",
            recommendation="Ajuste com amostras do caso, especialmente quando houver muita movimentação ou cortes bruscos.",
        )))
        return tab

    def _build_face_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._backend_input = QLineEdit(tab)
        self._model_name_input = QLineEdit(tab)
        self._ctx_id_spin = QSpinBox(tab)
        self._ctx_id_spin.setRange(-1, 64)
        self._minimum_face_quality_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._minimum_face_size_spin = QSpinBox(tab)
        self._minimum_face_size_spin.setRange(1, 4096)
        self._minimum_face_size_spin.setSuffix(" px")
        self._providers_input = QLineEdit(tab)
        self._providers_input.setPlaceholderText("Em branco = seleção automática")

        self._det_width_spin = QSpinBox(tab)
        self._det_width_spin.setRange(32, 4096)
        self._det_width_spin.setSingleStep(32)
        self._det_height_spin = QSpinBox(tab)
        self._det_height_spin.setRange(32, 4096)
        self._det_height_spin.setSingleStep(32)
        self._original_resolution_checkbox = QCheckBox("Manter resolução original do arquivo ou quadro", tab)
        self._original_resolution_checkbox.toggled.connect(self._det_width_spin.setDisabled)
        self._original_resolution_checkbox.toggled.connect(self._det_height_spin.setDisabled)
        det_size_widget = QWidget(tab)
        det_size_layout = QVBoxLayout(det_size_widget)
        det_size_layout.setContentsMargins(0, 0, 0, 0)
        det_row = QHBoxLayout()
        det_row.setContentsMargins(0, 0, 0, 0)
        det_row.addWidget(self._det_width_spin)
        det_row.addWidget(QLabel("x", tab))
        det_row.addWidget(self._det_height_spin)
        det_size_layout.addLayout(det_row)
        det_size_layout.addWidget(self._original_resolution_checkbox)

        form.addRow("Mecanismo:", self._with_help(self._backend_input, "Mecanismo", self._help_html(
            definition="Backend de análise facial responsável por detectar faces e extrair embeddings.",
            operational_effect="Define compatibilidade com modelos, dependência de hardware e comportamento da inferência.",
            recommendation="Mantenha insightface quando a estação estiver homologada para esse backend.",
            references=[("InsightFace", INSIGHTFACE_URL)],
        )))
        form.addRow("Modelo:", self._with_help(self._model_name_input, "Modelo", self._help_html(
            definition="Nome do conjunto de modelos carregado pelo backend facial.",
            operational_effect="Influencia robustez, velocidade, consumo de memória e resposta a pose, iluminação e resolução.",
            recommendation="Use buffalo_l como ponto de partida em estações compatíveis.",
            references=[("InsightFace", INSIGHTFACE_URL), ("SCRFD", SCRFD_URL)],
        )))
        form.addRow("Contexto de execução:", self._with_help(self._ctx_id_spin, "Contexto de execução", self._help_html(
            definition="Identificador do dispositivo usado pela inferência. Em geral, 0 indica o primeiro acelerador disponível e -1 força CPU.",
            operational_effect="Afeta desempenho, estabilidade do ambiente e reprodutibilidade operacional.",
            recommendation="Use -1 quando a prioridade for estabilidade e 0 apenas em hardware acelerado validado.",
            references=[("ONNX Runtime execution providers", ONNXRUNTIME_EP_URL)],
        )))
        form.addRow("Tamanho de detecção:", self._with_help(det_size_widget, "Tamanho de detecção", self._help_html(
            definition="Resolução operacional usada pelo detector facial antes da inferência, sem alterar o arquivo original.",
            operational_effect="Valores maiores tendem a ajudar em faces pequenas, mas aumentam tempo e memória.",
            recommendation="Use 640x640 como padrão inicial e só aumente quando houver perda comprovada de rostos pequenos.",
            caveat="Ao manter a resolução original, o detector usa a própria geometria do quadro analisado.",
            references=[("SCRFD", SCRFD_URL)],
        )))
        form.addRow("Qualidade mínima da face:", self._with_help(self._minimum_face_quality_spin, "Qualidade mínima da face", self._help_html(
            definition="Limiar mínimo da pontuação de detecção para que uma face siga no inventário.",
            operational_effect="Aumentar o limiar reduz ruído e falsos positivos, mas pode descartar rostos reais em condições adversas.",
            recommendation="Comece em 0,60 e calibre com amostras do próprio caso.",
            references=[("ArcFace", ARCFACE_URL)],
        )))
        form.addRow("Tamanho mínimo da face:", self._with_help(self._minimum_face_size_spin, "Tamanho mínimo da face", self._help_html(
            definition="Menor lado aceitável, em pixels, para a caixa delimitadora da face.",
            operational_effect="Valores baixos ampliam cobertura, mas aumentam recortes pouco informativos e ruído operacional.",
            recommendation="Use 40 px a 64 px como faixa inicial em triagem geral.",
        )))
        form.addRow("Mecanismos de execução:", self._with_help(self._providers_input, "Mecanismos de execução", self._help_html(
            definition="Lista, em ordem de preferência, dos execution providers do ONNX Runtime.",
            operational_effect="A ordem interfere em desempenho, compatibilidade e fallback entre GPU e CPU.",
            recommendation="Deixe em branco para seleção automática até que a infraestrutura esteja validada.",
            references=[("ONNX Runtime execution providers", ONNXRUNTIME_EP_URL)],
        )))
        return tab

    def _build_tracking_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._tracking_iou_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._tracking_spatial_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._tracking_embedding_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._tracking_min_total_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._tracking_geometry_weight_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._tracking_embedding_weight_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._tracking_max_missed_spin = QSpinBox(tab)
        self._tracking_max_missed_spin.setRange(0, 10_000)
        self._tracking_confidence_margin_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._tracking_representative_embeddings_spin = QSpinBox(tab)
        self._tracking_representative_embeddings_spin.setRange(1, 256)
        self._tracking_top_crops_spin = QSpinBox(tab)
        self._tracking_top_crops_spin.setRange(1, 256)
        self._tracking_quality_margin_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)

        form.addRow("IoU mínimo:", self._with_help(self._tracking_iou_spin, "IoU mínimo", self._help_html(
            definition="IoU mínimo para que uma detecção seja candidata a continuar um track ativo.",
            operational_effect="Valores maiores exigem continuidade geométrica mais forte; valores menores toleram mais variação espacial.",
            recommendation="Mantenha valor baixo, mas não nulo, para não fragmentar tracks por pequenas oscilações."
        )))
        form.addRow("Distância espacial máxima:", self._with_help(self._tracking_spatial_spin, "Distância espacial máxima", self._help_html(
            definition="Distância normalizada máxima entre centros das caixas para permitir associação temporal.",
            operational_effect="Controla quanto uma face pode se deslocar entre amostras sem abrir novo track.",
            recommendation="Aumente com cautela quando o vídeo tiver muita movimentação ou intervalo de amostragem mais espaçado."
        )))
        form.addRow("Similaridade mínima de embedding:", self._with_help(self._tracking_embedding_spin, "Similaridade mínima de embedding", self._help_html(
            definition="Limiar de similaridade usado para reforçar a associação entre detecção e track quando a geometria não basta.",
            operational_effect="Ajuda a desambiguar múltiplas faces ativas ao mesmo tempo.",
            recommendation="Ajuste em conjunto com IoU, distância espacial e pontuação total mínima."
        )))
        form.addRow("Pontuação total mínima:", self._with_help(self._tracking_min_total_spin, "Pontuação total mínima", self._help_html(
            definition="Pontuação composta mínima para manter uma detecção no mesmo track em vez de iniciar outro.",
            operational_effect="Valores mais altos reduzem fusões indevidas, mas podem fragmentar tracks persistentes.",
            recommendation="Calibre com vídeos que tenham múltiplas faces e mudanças de pose."
        )))
        form.addRow("Peso geométrico:", self._with_help(self._tracking_geometry_weight_spin, "Peso geométrico", self._help_html(
            definition="Peso relativo da componente geométrica na pontuação total de associação.",
            operational_effect="Valores mais altos priorizam continuidade espacial; valores menores privilegiam embedding.",
            recommendation="Mantenha a soma dos pesos geométrico e de embedding próxima de 1."
        )))
        form.addRow("Peso do embedding:", self._with_help(self._tracking_embedding_weight_spin, "Peso do embedding", self._help_html(
            definition="Peso relativo da similaridade de embedding na pontuação total de associação.",
            operational_effect="Aumenta a influência do vetor facial na continuidade do track.",
            recommendation="Suba esse peso quando houver muitas faces próximas com movimento semelhante."
        )))
        form.addRow("Perdas máximas consecutivas:", self._with_help(self._tracking_max_missed_spin, "Perdas máximas consecutivas", self._help_html(
            definition="Número máximo de amostras seguidas sem correspondência antes de encerrar um track.",
            operational_effect="Valores maiores toleram oclusões curtas, mas também podem estender tracks indevidamente.",
            recommendation="Ajuste conforme o intervalo de amostragem e a frequência de oclusões do material."
        )))
        form.addRow("Margem de confiança:", self._with_help(self._tracking_confidence_margin_spin, "Margem de confiança", self._help_html(
            definition="Diferença mínima entre o melhor e o segundo melhor candidato para aceitar a associação.",
            operational_effect="Evita associações ambíguas quando duas faces competem pelo mesmo track.",
            recommendation="Aumente esse valor em cenas densas com várias pessoas próximas."
        )))
        form.addRow("Embeddings representativos por track:", self._with_help(self._tracking_representative_embeddings_spin, "Embeddings representativos por track", self._help_html(
            definition="Quantidade máxima de embeddings de referência preservados por track.",
            operational_effect="Mais embeddings melhoram representatividade intra-track, mas aumentam custo e memória.",
            recommendation="Use valor moderado, como 3 a 8, salvo necessidade de maior variabilidade."
        )))
        form.addRow("Melhores recortes por track:", self._with_help(self._tracking_top_crops_spin, "Melhores recortes por track", self._help_html(
            definition="Quantidade de recortes de melhor qualidade mantidos como representação visual do track.",
            operational_effect="Aumenta material auditável por track, mas também o volume de artefatos gerados.",
            recommendation="Use poucos recortes de alta qualidade para manter o relatório legível."
        )))
        form.addRow("Ganho mínimo de qualidade para novo pico:", self._with_help(self._tracking_quality_margin_spin, "Ganho mínimo de qualidade para novo pico", self._help_html(
            definition="Diferença mínima de qualidade para considerar que uma nova detecção supera o melhor estado atual do track.",
            operational_effect="Influencia a seleção de keyframes por melhora de qualidade.",
            recommendation="Use margem pequena para capturar evolução gradual sem gerar excesso de keyframes."
        )))
        return tab

    def _build_clustering_search_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._assignment_similarity_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._candidate_similarity_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._min_cluster_size_spin = QSpinBox(tab)
        self._min_cluster_size_spin.setRange(1, 10_000)
        self._min_track_size_spin = QSpinBox(tab)
        self._min_track_size_spin.setRange(1, 10_000)
        self._search_enabled_checkbox = QCheckBox("Habilitar indexação vetorial", tab)
        self._search_prefer_faiss_checkbox = QCheckBox("Preferir FAISS quando disponível", tab)
        self._search_coarse_top_k_spin = QSpinBox(tab)
        self._search_coarse_top_k_spin.setRange(1, 10_000)
        self._search_refine_top_k_spin = QSpinBox(tab)
        self._search_refine_top_k_spin.setRange(1, 10_000)

        form.addRow("Limiar de atribuição:", self._with_help(self._assignment_similarity_spin, "Limiar de atribuição", self._help_html(
            definition="Similaridade mínima para fundir tracks no mesmo grupo.",
            operational_effect="Valores altos reduzem risco de fusão indevida, mas tendem a fragmentar a saída.",
            recommendation="Comece entre 0,50 e 0,60 e ajuste após revisão visual dos grupos."
        )))
        form.addRow("Limiar de sugestão entre grupos:", self._with_help(self._candidate_similarity_spin, "Limiar de sugestão entre grupos", self._help_html(
            definition="Similaridade usada para apontar grupos distintos como correlação provável a revisar.",
            operational_effect="Funciona como zona de atenção; não provoca fusão automática.",
            recommendation="Mantenha abaixo do limiar de atribuição para separar agrupamento de revisão investigativa."
        )))
        form.addRow("Tamanho mínimo do grupo:", self._with_help(self._min_cluster_size_spin, "Tamanho mínimo do grupo", self._help_html(
            definition="Quantidade mínima de tracks para um grupo aparecer no resultado final.",
            operational_effect="Valores maiores reduzem grupos pequenos e ruído visual, mas podem suprimir indivíduos raros.",
            recommendation="Use 1 quando a prioridade for completude do inventário."
        )))
        form.addRow("Tamanho mínimo do track:", self._with_help(self._min_track_size_spin, "Tamanho mínimo do track", self._help_html(
            definition="Quantidade mínima de ocorrências para que um track participe do clustering.",
            operational_effect="Ajuda a descartar tracks muito curtos ou pouco estáveis antes do agrupamento.",
            recommendation="Aumente somente quando houver excesso de tracks unitários sem valor analítico."
        )))
        form.addRow("Busca vetorial:", self._with_help(self._search_enabled_checkbox, "Busca vetorial", self._help_html(
            definition="Ativa a criação do índice vetorial coarse-to-fine para tracks e grupos.",
            operational_effect="Amplia capacidade de busca posterior por indivíduos e grupos semelhantes.",
            recommendation="Mantenha habilitado quando houver interesse em pesquisa ou revisão posterior de ocorrências."
        )))
        form.addRow("Preferência por FAISS:", self._with_help(self._search_prefer_faiss_checkbox, "Preferência por FAISS", self._help_html(
            definition="Prioriza FAISS como engine do índice vetorial quando o pacote estiver instalado.",
            operational_effect="Tende a melhorar desempenho e escalabilidade em conjuntos maiores.",
            recommendation="Deixe ativo quando a estação já tiver o pacote validado.",
            references=[("FAISS", FAISS_URL)]
        )))
        form.addRow("Top-K coarse:", self._with_help(self._search_coarse_top_k_spin, "Top-K coarse", self._help_html(
            definition="Quantidade de grupos candidatos recuperados na etapa coarse da busca.",
            operational_effect="Valores maiores ampliam cobertura da busca, mas aumentam custo do refinamento.",
            recommendation="Ajuste conforme o volume de grupos e a necessidade de recall."
        )))
        form.addRow("Top-K refinado:", self._with_help(self._search_refine_top_k_spin, "Top-K refinado", self._help_html(
            definition="Quantidade de tracks e ocorrências mantidos após o refinamento interno.",
            operational_effect="Controla quantos resultados detalhados serão examinados na etapa final.",
            recommendation="Use valor um pouco maior que o coarse quando quiser expandir a inspeção dentro dos grupos candidatos."
        )))
        return tab

    def _build_enhancement_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._enhancement_enable_checkbox = QCheckBox("Habilitar pré-processamento auditável", tab)
        self._enhancement_min_brightness_spin = self._double_spin_box(0.0, 1.0, 3, 0.01)
        self._enhancement_clahe_clip_spin = self._double_spin_box(0.1, 100.0, 2, 0.1)
        self._enhancement_clahe_tile_spin = QSpinBox(tab)
        self._enhancement_clahe_tile_spin.setRange(1, 128)
        self._enhancement_gamma_spin = self._double_spin_box(0.1, 5.0, 2, 0.05)
        self._enhancement_denoise_spin = QSpinBox(tab)
        self._enhancement_denoise_spin.setRange(0, 100)

        form.addRow("Pré-processamento:", self._with_help(self._enhancement_enable_checkbox, "Pré-processamento", self._help_html(
            definition="Ativa o fluxo de melhoria auditável em cenários de baixa iluminação ou ruído.",
            operational_effect="Pode melhorar detecção e qualidade dos recortes, preservando metadados do que foi aplicado.",
            recommendation="Mantenha ativo em triagens de campo ou acervos heterogêneos; desative apenas em comparações estritamente controladas."
        )))
        form.addRow("Brilho mínimo para aprimorar:", self._with_help(self._enhancement_min_brightness_spin, "Brilho mínimo para aprimorar", self._help_html(
            definition="Abaixo desse brilho médio normalizado, o sistema considera aplicar CLAHE e gamma.",
            operational_effect="Determina o gatilho para compensar material subexposto.",
            recommendation="Calibre com frames realmente escuros do caso para evitar processamento desnecessário."
        )))
        form.addRow("CLAHE clip limit:", self._with_help(self._enhancement_clahe_clip_spin, "CLAHE clip limit", self._help_html(
            definition="Intensidade do contraste local aplicado pelo CLAHE.",
            operational_effect="Valores altos aumentam contraste, mas podem realçar ruído e artefatos.",
            recommendation="Use valor moderado e registre no relatório qualquer calibração fora do padrão."
        )))
        form.addRow("CLAHE tile grid:", self._with_help(self._enhancement_clahe_tile_spin, "CLAHE tile grid", self._help_html(
            definition="Tamanho da malha usada pelo CLAHE para ajuste local de contraste.",
            operational_effect="Afeta granularidade do contraste local aplicado na imagem.",
            recommendation="Mantenha o padrão salvo necessidade específica demonstrada em validação."
        )))
        form.addRow("Gamma:", self._with_help(self._enhancement_gamma_spin, "Gamma", self._help_html(
            definition="Correção gama aplicada como parte do pré-processamento.",
            operational_effect="Pode clarear ou escurecer a imagem antes da detecção facial.",
            recommendation="Evite valores extremos para não distorcer a aparência do material."
        )))
        form.addRow("Denoise:", self._with_help(self._enhancement_denoise_spin, "Denoise", self._help_html(
            definition="Força do filtro de redução de ruído aplicado no pré-processamento.",
            operational_effect="Ajuda em material ruidoso, mas valores altos podem suavizar demais detalhes faciais.",
            recommendation="Use 0 quando o ruído não for problema relevante e suba gradualmente em validação."
        )))
        return tab

    def _build_reporting_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._compile_pdf_checkbox = QCheckBox("Compilar PDF automaticamente", tab)
        self._max_tracks_per_group_spin = QSpinBox(tab)
        self._max_tracks_per_group_spin.setRange(1, 10_000)
        self._chain_note_input = QPlainTextEdit(tab)
        self._chain_note_input.setMinimumHeight(150)
        self._chain_note_input.setPlaceholderText("Informe a nota de cadeia de custódia.")

        form.addRow("Compilação do PDF:", self._with_help(self._compile_pdf_checkbox, "Compilação do PDF", self._help_html(
            definition="Define se o TEX será convertido automaticamente em PDF ao final da execução.",
            operational_effect="Entrega o documento final pronto para revisão quando a estação tiver LaTeX funcional.",
            recommendation="Mantenha ativo em produção e desative apenas durante depuração ou revisão do TEX."
        )))
        form.addRow("Tracks por grupo no relatório:", self._with_help(self._max_tracks_per_group_spin, "Tracks por grupo no relatório", self._help_html(
            definition="Quantidade máxima de tracks detalhados por grupo na seção de resultados.",
            operational_effect="Controla densidade visual e tamanho do relatório.",
            recommendation="Use valor moderado para manter o documento legível sem perder representatividade."
        )))
        form.addRow("Nota de cadeia de custódia:", self._with_help(self._chain_note_input, "Nota de cadeia de custódia", self._help_html(
            definition="Texto institucional inserido no relatório para registrar preservação dos originais, segregação dos derivados e caráter probabilístico do processamento.",
            operational_effect="Ajuda a alinhar o documento técnico ao procedimento oficial do órgão.",
            recommendation="Padronize a redação segundo o procedimento institucional e declare explicitamente que não há prova conclusiva de identidade."
        )))
        return tab

    def _load_config(self, config: AppConfig) -> None:
        self._output_directory_input.setText(config.app.output_directory_name)
        self._report_title_input.setText(config.app.report_title)
        self._organization_input.setText(config.app.organization)
        self._log_level_combo.setCurrentText(config.app.log_level.upper())

        self._image_extensions_input.setText(", ".join(config.media.image_extensions))
        self._video_extensions_input.setText(", ".join(config.media.video_extensions))
        self._sampling_interval_spin.setValue(config.video.sampling_interval_seconds)
        self._keyframe_interval_spin.setValue(config.video.keyframe_interval_seconds)
        self._significant_change_spin.setValue(config.video.significant_change_threshold)
        unlimited_frames = config.video.max_frames_per_video is None
        self._unlimited_frames_checkbox.setChecked(unlimited_frames)
        self._max_frames_spin.setDisabled(unlimited_frames)
        self._max_frames_spin.setValue(config.video.max_frames_per_video or 1)

        self._backend_input.setText(config.face_model.backend)
        self._model_name_input.setText(config.face_model.model_name)
        self._ctx_id_spin.setValue(config.face_model.ctx_id)
        use_original_resolution = config.face_model.det_size is None
        self._original_resolution_checkbox.setChecked(use_original_resolution)
        det_size = config.face_model.det_size or (640, 640)
        self._det_width_spin.setValue(det_size[0])
        self._det_height_spin.setValue(det_size[1])
        self._det_width_spin.setDisabled(use_original_resolution)
        self._det_height_spin.setDisabled(use_original_resolution)
        self._minimum_face_quality_spin.setValue(config.face_model.minimum_face_quality)
        self._minimum_face_size_spin.setValue(config.face_model.minimum_face_size_pixels)
        self._providers_input.setText(", ".join(config.face_model.providers))

        self._tracking_iou_spin.setValue(config.tracking.iou_threshold)
        self._tracking_spatial_spin.setValue(config.tracking.spatial_distance_threshold)
        self._tracking_embedding_spin.setValue(config.tracking.embedding_similarity_threshold)
        self._tracking_min_total_spin.setValue(config.tracking.minimum_total_match_score)
        self._tracking_geometry_weight_spin.setValue(config.tracking.geometry_weight)
        self._tracking_embedding_weight_spin.setValue(config.tracking.embedding_weight)
        self._tracking_max_missed_spin.setValue(config.tracking.max_missed_detections)
        self._tracking_confidence_margin_spin.setValue(config.tracking.confidence_margin)
        self._tracking_representative_embeddings_spin.setValue(config.tracking.representative_embeddings_per_track)
        self._tracking_top_crops_spin.setValue(config.tracking.top_crops_per_track)
        self._tracking_quality_margin_spin.setValue(config.tracking.quality_improvement_margin)

        self._assignment_similarity_spin.setValue(config.clustering.assignment_similarity)
        self._candidate_similarity_spin.setValue(config.clustering.candidate_similarity)
        self._min_cluster_size_spin.setValue(config.clustering.min_cluster_size)
        self._min_track_size_spin.setValue(config.clustering.min_track_size)
        self._search_enabled_checkbox.setChecked(config.search.enabled)
        self._search_prefer_faiss_checkbox.setChecked(config.search.prefer_faiss)
        self._search_coarse_top_k_spin.setValue(config.search.coarse_top_k)
        self._search_refine_top_k_spin.setValue(config.search.refine_top_k)

        self._enhancement_enable_checkbox.setChecked(config.enhancement.enable_preprocessing)
        self._enhancement_min_brightness_spin.setValue(config.enhancement.minimum_brightness_to_enhance)
        self._enhancement_clahe_clip_spin.setValue(config.enhancement.clahe_clip_limit)
        self._enhancement_clahe_tile_spin.setValue(config.enhancement.clahe_tile_grid_size)
        self._enhancement_gamma_spin.setValue(config.enhancement.gamma)
        self._enhancement_denoise_spin.setValue(config.enhancement.denoise_strength)

        self._compile_pdf_checkbox.setChecked(config.reporting.compile_pdf)
        self._max_tracks_per_group_spin.setValue(config.reporting.max_tracks_per_group)
        self._chain_note_input.setPlainText(config.forensics.chain_of_custody_note)

    def _restore_selected_config(self) -> None:
        self._load_config(self._selected_config)

    def _accept_configuration(self) -> None:
        try:
            self._selected_config = self._build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Configuração inválida", str(exc))
            return
        self.accept()

    def _build_config(self) -> AppConfig:
        output_directory_name = self._require_text(self._output_directory_input.text(), "Informe o diretório de saída da execução.")
        report_title = self._require_text(self._report_title_input.text(), "Informe o título do relatório.")
        organization = self._require_text(self._organization_input.text(), "Informe a organização responsável.")
        backend = self._require_text(self._backend_input.text(), "Informe o mecanismo facial.")
        model_name = self._require_text(self._model_name_input.text(), "Informe o nome do modelo facial.")
        chain_note = self._require_text(self._chain_note_input.toPlainText(), "Informe a nota de cadeia de custódia.")

        geometry_weight = float(self._tracking_geometry_weight_spin.value())
        embedding_weight = float(self._tracking_embedding_weight_spin.value())
        if geometry_weight == 0.0 and embedding_weight == 0.0:
            raise ValueError("Ao menos um dos pesos do tracking deve ser maior que zero.")

        return AppConfig(
            app=AppSettings(
                name=self._selected_config.app.name,
                output_directory_name=output_directory_name,
                report_title=report_title,
                organization=organization,
                log_level=self._log_level_combo.currentText(),
                mediainfo_directory=None,
            ),
            media=MediaSettings(
                image_extensions=self._parse_extensions(self._image_extensions_input.text(), "imagem"),
                video_extensions=self._parse_extensions(self._video_extensions_input.text(), "vídeo"),
            ),
            video=VideoSettings(
                sampling_interval_seconds=float(self._sampling_interval_spin.value()),
                max_frames_per_video=(None if self._unlimited_frames_checkbox.isChecked() else int(self._max_frames_spin.value())),
                keyframe_interval_seconds=float(self._keyframe_interval_spin.value()),
                significant_change_threshold=float(self._significant_change_spin.value()),
            ),
            face_model=FaceModelSettings(
                backend=backend,
                model_name=model_name,
                det_size=(None if self._original_resolution_checkbox.isChecked() else (int(self._det_width_spin.value()), int(self._det_height_spin.value()))),
                minimum_face_quality=float(self._minimum_face_quality_spin.value()),
                minimum_face_size_pixels=int(self._minimum_face_size_spin.value()),
                ctx_id=int(self._ctx_id_spin.value()),
                providers=self._parse_list(self._providers_input.text()),
            ),
            clustering=ClusteringSettings(
                assignment_similarity=float(self._assignment_similarity_spin.value()),
                candidate_similarity=float(self._candidate_similarity_spin.value()),
                min_cluster_size=int(self._min_cluster_size_spin.value()),
                min_track_size=int(self._min_track_size_spin.value()),
            ),
            reporting=ReportingSettings(
                compile_pdf=self._compile_pdf_checkbox.isChecked(),
                max_tracks_per_group=int(self._max_tracks_per_group_spin.value()),
            ),
            forensics=ForensicsSettings(chain_of_custody_note=chain_note),
            tracking=TrackingSettings(
                iou_threshold=float(self._tracking_iou_spin.value()),
                spatial_distance_threshold=float(self._tracking_spatial_spin.value()),
                embedding_similarity_threshold=float(self._tracking_embedding_spin.value()),
                minimum_total_match_score=float(self._tracking_min_total_spin.value()),
                geometry_weight=geometry_weight,
                embedding_weight=embedding_weight,
                max_missed_detections=int(self._tracking_max_missed_spin.value()),
                confidence_margin=float(self._tracking_confidence_margin_spin.value()),
                representative_embeddings_per_track=int(self._tracking_representative_embeddings_spin.value()),
                top_crops_per_track=int(self._tracking_top_crops_spin.value()),
                quality_improvement_margin=float(self._tracking_quality_margin_spin.value()),
            ),
            enhancement=EnhancementSettings(
                enable_preprocessing=self._enhancement_enable_checkbox.isChecked(),
                minimum_brightness_to_enhance=float(self._enhancement_min_brightness_spin.value()),
                clahe_clip_limit=float(self._enhancement_clahe_clip_spin.value()),
                clahe_tile_grid_size=int(self._enhancement_clahe_tile_spin.value()),
                gamma=float(self._enhancement_gamma_spin.value()),
                denoise_strength=int(self._enhancement_denoise_spin.value()),
            ),
            search=SearchSettings(
                enabled=self._search_enabled_checkbox.isChecked(),
                prefer_faiss=self._search_prefer_faiss_checkbox.isChecked(),
                coarse_top_k=int(self._search_coarse_top_k_spin.value()),
                refine_top_k=int(self._search_refine_top_k_spin.value()),
            ),
        )

    def _with_help(self, widget: QWidget, title: str, body: str) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)

        help_button = QToolButton(container)
        help_button.setText("?")
        help_button.setToolTip("Ajuda técnica")
        help_button.setFocusPolicy(Qt.NoFocus)
        help_button.clicked.connect(lambda _checked=False, t=title, b=body: self._show_help(t, b))
        layout.addWidget(help_button)
        return container

    def _with_directory_selector(self, line_edit: QLineEdit, caption: str) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        browse_button = QPushButton("Selecionar...", container)
        browse_button.clicked.connect(lambda _checked=False, target=line_edit, dialog_caption=caption: self._select_directory(target, dialog_caption))
        layout.addWidget(browse_button)
        return container

    def _select_directory(self, target: QLineEdit, caption: str) -> None:
        initial_directory = target.text().strip() or ""
        selected_directory = QFileDialog.getExistingDirectory(self, caption, initial_directory)
        if selected_directory:
            target.setText(selected_directory)

    def _show_help(self, title: str, body: str) -> None:
        self._help_title_label.setText(title)
        self._help_body.setHtml(body)

    def _help_html(
        self,
        *,
        definition: str,
        operational_effect: str,
        recommendation: str,
        caveat: str | None = None,
        references: list[tuple[str, str]] | None = None,
    ) -> str:
        reference_items = references or []
        references_html = "".join(f"<li>{self._abnt_reference_html(label, url)}</li>" for label, url in reference_items)
        caveat_html = f"<p><b>Observação.</b> {caveat}</p>" if caveat else ""
        references_block = f"<p><b>Referências.</b></p><ul>{references_html}</ul>" if reference_items else ""
        return (
            "<div style='font-family: Segoe UI, sans-serif; font-size: 10pt;'>"
            f"<p><b>Definição.</b> {definition}</p>"
            f"<p><b>Efeito operacional.</b> {operational_effect}</p>"
            f"<p><b>Recomendação técnica.</b> {recommendation}</p>"
            f"{caveat_html}"
            f"{references_block}"
            "</div>"
        )

    def _abnt_reference_html(self, label: str, url: str) -> str:
        entries = {
            INSIGHTFACE_URL: (
                "INSIGHTFACE. <i>InsightFace: an open source 2D and 3D deep face analysis library</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            ARCFACE_URL: (
                "DENG, Jiankang et al. <i>ArcFace: additive angular margin loss for deep face recognition</i>. "
                "In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            SCRFD_URL: (
                "GUO, Jia et al. <i>Sample and computation redistribution for efficient face detection</i>. "
                f"arXiv, 2021. Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            ONNXRUNTIME_EP_URL: (
                "MICROSOFT. <i>ONNX Runtime: execution providers</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            OPENCV_VIDEOCAPTURE_URL: (
                "OPENCV. <i>OpenCV 4.x documentation: VideoCapture class reference</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            MEDIAINFO_URL: (
                "MEDIAAREA. <i>MediaInfo</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            FAISS_URL: (
                "META AI. <i>FAISS</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
        }
        return entries.get(
            url,
            f"{label}. Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}",
        )

    def _double_spin_box(self, minimum: float, maximum: float, decimals: int, step: float, suffix: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    def _parse_extensions(self, raw_value: str, label: str) -> tuple[str, ...]:
        extensions = []
        for item in self._parse_list(raw_value):
            normalized = item.lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            extensions.append(normalized)
        if not extensions:
            raise ValueError(f"Informe ao menos uma extensão de {label}.")
        return tuple(dict.fromkeys(extensions))

    def _parse_list(self, raw_value: str) -> tuple[str, ...]:
        parts = [item.strip() for item in re.split(r"[,;\r\n]+", raw_value) if item.strip()]
        return tuple(dict.fromkeys(parts))

    def _require_text(self, value: str, error_message: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(error_message)
        return normalized

    def _optional_directory(self, value: str, label: str) -> str | None:
        normalized = value.strip()
        if not normalized:
            return None
        candidate = Path(normalized).expanduser()
        if not candidate.exists():
            raise ValueError(f"O diretório de {label} informado não existe.")
        if not candidate.is_dir():
            raise ValueError(f"Informe a pasta que contém o executável do {label}.")
        return str(candidate.resolve())
