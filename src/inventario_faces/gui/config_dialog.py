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
    FaceModelSettings,
    ForensicsSettings,
    MediaSettings,
    ReportingSettings,
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
OPENCV_IMGCODECS_URL = "https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html"
NIST_FRVT_URL = "https://www.nist.gov/system/files/documents/2019/12/03/FRVT_ongoing_1N_api_v1.0.pdf"
NIST_CHAIN_OF_CUSTODY_URL = "https://csrc.nist.gov/glossary/term/chain_of_custody"
NIST_CFREDS_URL = "https://cfreds.nist.gov/"
MEDIAINFO_URL = "https://mediaarea.net/en/MediaInfo"
ABNT_ACCESS_DATE = "Acesso em: 3 abr. 2026."

PT_REPLACEMENTS = {
    "Ajuste os parametros usados na varredura, na analise facial, no agrupamento e na emissao do relatorio. As alteracoes aplicadas ficam persistidas para as proximas execucoes.": (
        "Ajuste os parâmetros usados na varredura, na análise facial, no agrupamento e na emissão do relatório. "
        "As alterações aplicadas ficam persistidas para as próximas execuções."
    ),
    "Clique em um dos botoes ? ao lado de cada parametro para ver a descricao detalhada do campo selecionado.": (
        "Clique em um dos botões ? ao lado de cada parâmetro para ver a descrição detalhada do campo selecionado."
    ),
    "Cada ajuda apresenta o significado do parametro, o impacto esperado na execucao e os riscos operacionais mais comuns quando ele e ajustado sem calibracao.": (
        "Cada ajuda apresenta o significado do parâmetro, o impacto esperado na execução e os riscos operacionais mais comuns quando ele é ajustado sem calibração."
    ),
    "Revise as configuracoes antes de processar um novo acervo e mantenha, no relatorio tecnico, a coerencia entre os parametros usados e a finalidade do exame.": (
        "Revise as configurações antes de processar um novo acervo e mantenha, no relatório técnico, a coerência entre os parâmetros usados e a finalidade do exame."
    ),
    "Ajuda tecnica": "Ajuda técnica",
    "Configuracoes do procedimento": "Configurações do procedimento",
    "Restaurar valores carregados": "Restaurar valores carregados",
    "Aplicar": "Aplicar",
    "Cancelar": "Cancelar",
    "Geral": "Geral",
    "Midias": "Mídias",
    "Analise facial": "Análise facial",
    "Agrupamento": "Agrupamento",
    "Relatorio": "Relatório",
    "Diretorio de saida": "Diretório de saída",
    "Diretorio de saida:": "Diretório de saída:",
    "Titulo do relatorio": "Título do relatório",
    "Titulo do relatorio:": "Título do relatório:",
    "Organizacao responsavel": "Organização responsável",
    "Organizacao responsavel:": "Organização responsável:",
    "Nivel de log": "Nível de log",
    "Nivel de log:": "Nível de log:",
    "Extensoes de imagem": "Extensões de imagem",
    "Extensoes de imagem:": "Extensões de imagem:",
    "Extensoes de video": "Extensões de vídeo",
    "Extensoes de video:": "Extensões de vídeo:",
    "Intervalo de amostragem": "Intervalo de amostragem",
    "Intervalo de amostragem:": "Intervalo de amostragem:",
    "Maximo de quadros por video": "Máximo de quadros por vídeo",
    "Maximo de quadros por video:": "Máximo de quadros por vídeo:",
    "Sem limite de quadros por video": "Sem limite de quadros por vídeo",
    "Mecanismo": "Mecanismo",
    "Modelo": "Modelo",
    "Contexto": "Contexto",
    "Tamanho de deteccao": "Tamanho de detecção",
    "Tamanho de deteccao:": "Tamanho de detecção:",
    "Qualidade minima da face": "Qualidade mínima da face",
    "Qualidade minima da face:": "Qualidade mínima da face:",
    "Tamanho minimo da face": "Tamanho mínimo da face",
    "Tamanho minimo da face:": "Tamanho mínimo da face:",
    "Mecanismos de execucao": "Mecanismos de execução",
    "Mecanismos de execucao:": "Mecanismos de execução:",
    "Limiar de atribuicao": "Limiar de atribuição",
    "Limiar de atribuicao:": "Limiar de atribuição:",
    "Limiar de sugestao entre grupos": "Limiar de sugestão entre grupos",
    "Limiar de sugestao entre grupos:": "Limiar de sugestão entre grupos:",
    "Tamanho minimo do grupo": "Tamanho mínimo do grupo",
    "Tamanho minimo do grupo:": "Tamanho mínimo do grupo:",
    "Faces maximas por individuo na galeria": "Faces máximas por indivíduo na galeria",
    "Faces maximas por individuo na galeria:": "Faces máximas por indivíduo na galeria:",
    "Compilacao do PDF": "Compilação do PDF",
    "Compilacao do PDF:": "Compilação do PDF:",
    "Nota de cadeia de custodia": "Nota de cadeia de custódia",
    "Nota de cadeia de custodia:": "Nota de cadeia de custódia:",
    "Em branco = selecao automatica": "Em branco = seleção automática",
    "Informe a nota de cadeia de custodia.": "Informe a nota de cadeia de custódia.",
    "Observacao.": "Observação.",
    "Definicao.": "Definição.",
    "Efeito operacional.": "Efeito operacional.",
    "Recomendacao tecnica.": "Recomendação técnica.",
    "Referencias.": "Referências.",
    "analise": "análise",
    "Analise": "Análise",
    "emissao": "emissão",
    "Emissao": "Emissão",
    "relatorio": "relatório",
    "Relatorio": "Relatório",
    "configuracoes": "configurações",
    "Configuracoes": "Configurações",
    "parametro": "parâmetro",
    "parametros": "parâmetros",
    "proximas": "próximas",
    "execucao": "execução",
    "execucoes": "execuções",
    "descricao": "descrição",
    "tecnico": "técnico",
    "tecnica": "técnica",
    "coerencia": "coerência",
    "botao": "botão",
    "botoes": "botões",
    "detalhada": "detalhada",
    "significado": "significado",
    "calibracao": "calibração",
    "Diretorio": "Diretório",
    "saida": "saída",
    "Titulo": "Título",
    "Organizacao": "Organização",
    "responsavel": "responsável",
    "Nivel": "Nível",
    "Midia": "Mídia",
    "midia": "mídia",
    "Midias": "Mídias",
    "midias": "mídias",
    "video": "vídeo",
    "Video": "Vídeo",
    "videos": "vídeos",
    "Videos": "Vídeos",
    "Extensoes": "Extensões",
    "extensoes": "extensões",
    "classificacao": "classificação",
    "evidencias": "evidências",
    "variacao": "variação",
    "deteccao": "detecção",
    "selecao": "seleção",
    "atribuicao": "atribuição",
    "sugestao": "sugestão",
    "dimensao": "dimensão",
    "minimo": "mínimo",
    "maximo": "máximo",
    "media": "média",
    "padrao": "padrão",
    "informacao": "informação",
    "informacoes": "informações",
    "caixa delimitadora": "caixa delimitadora",
    "pontuacao": "pontuação",
    "revisao": "revisão",
    "tambem": "também",
    "possiveis": "possíveis",
    "individuos": "indivíduos",
    "execucao": "execução",
    "reproducibilidade": "reprodutibilidade",
    "originais": "originais",
    "documentacao": "documentação",
    "institucional": "institucional",
    "identificacao": "identificação",
    "probabilistica": "probabilística",
    "administrativa": "administrativa",
    "unidade": "unidade",
    "diagnostico": "diagnóstico",
    "homologacao": "homologação",
    "operacao": "operação",
    "restrita": "restrita",
    "temporal": "temporal",
    "exploratorias": "exploratórias",
    "justificativa": "justificativa",
    "metodologica": "metodológica",
    "integral": "integral",
    "infraestrutura": "infraestrutura",
    "vetores": "vetores",
    "caracteristicas": "características",
    "dependencia": "dependência",
    "ecossistema": "ecossistema",
    "variacoes": "variações",
    "memoria": "memória",
    "compativeis": "compatíveis",
    "homologada": "homologada",
    "validacao": "validação",
    "documental": "documental",
    "dispositivo": "dispositivo",
    "acelerado": "acelerado",
    "disponivel": "disponível",
    "forca": "força",
    "grafo": "grafo",
    "reprodutivel": "reprodutível",
    "Resolucao": "Resolução",
    "resolucao": "resolução",
    "inferencia": "inferência",
    "informativos": "informativos",
    "ruido": "ruído",
    "privilegio": "privilégio",
    "abrangencia": "abrangência",
    "nao": "não",
    "criterio": "critério",
    "automatica": "automática",
    "automatico": "automático",
    "estacoes": "estações",
    "estacao": "estação",
    "ja": "já",
    "validada": "validada",
    "correlacao": "correlação",
    "indicio": "indício",
    "identidade": "identidade",
    "permanece": "permanece",
    "ocorrencias": "ocorrências",
    "completude": "completude",
    "depuracao": "depuração",
    "custodia": "custódia",
    "vestigios": "vestígios",
    "carater": "caráter",
}


def pt(text: str) -> str:
    normalized = text
    for source, target in PT_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


class ConfigDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_config = config
        self.setWindowTitle(pt("Configuracoes do procedimento"))
        self.resize(860, 820)
        self._build_ui()
        self._load_config(config)

    @property
    def selected_config(self) -> AppConfig:
        return self._selected_config

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        description = QLabel(
            pt(
                "Ajuste os parametros usados na varredura, na analise facial, no agrupamento e na "
                "emissao do relatorio. As alteracoes aplicadas ficam persistidas para as proximas execucoes."
            )
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        tabs = QTabWidget(self)
        tabs.addTab(self._build_general_tab(), pt("Geral"))
        tabs.addTab(self._build_media_tab(), pt("Midias"))
        tabs.addTab(self._build_face_tab(), pt("Analise facial"))
        tabs.addTab(self._build_clustering_tab(), pt("Agrupamento"))
        tabs.addTab(self._build_reporting_tab(), pt("Relatorio"))
        layout.addWidget(tabs)

        self._help_title_label = QLabel(pt("Ajuda tecnica"))
        self._help_title_label.setStyleSheet("font-weight: bold;")
        self._help_body = QTextBrowser(self)
        self._help_body.setReadOnly(True)
        self._help_body.setOpenExternalLinks(True)
        self._help_body.setMinimumHeight(210)
        self._help_body.setHtml(
            self._help_html(
                definition=(
                    "Clique em um dos botoes ? ao lado de cada parametro para ver a descricao detalhada "
                    "do campo selecionado."
                ),
                operational_effect=(
                    "Cada ajuda apresenta o significado do parametro, o impacto esperado na execucao "
                    "e os riscos operacionais mais comuns quando ele e ajustado sem calibracao."
                ),
                recommendation=(
                    "Revise as configuracoes antes de processar um novo acervo e mantenha, no relatorio "
                    "tecnico, a coerencia entre os parametros usados e a finalidade do exame."
                ),
                references=[
                    ("InsightFace (repositorio oficial)", INSIGHTFACE_URL),
                    ("ONNX Runtime Execution Providers", ONNXRUNTIME_EP_URL),
                    ("OpenCV VideoCapture", OPENCV_VIDEOCAPTURE_URL),
                    ("NIST FRVT 1:N API", NIST_FRVT_URL),
                ],
            )
        )
        layout.addWidget(self._help_title_label)
        layout.addWidget(self._help_body)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        apply_button = button_box.button(QDialogButtonBox.Ok)
        cancel_button = button_box.button(QDialogButtonBox.Cancel)
        if apply_button is not None:
            apply_button.setText(pt("Aplicar"))
        if cancel_button is not None:
            cancel_button.setText(pt("Cancelar"))

        restore_button = QPushButton(pt("Restaurar valores carregados"), self)
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
        self._mediainfo_directory_input = QLineEdit(tab)
        self._mediainfo_directory_input.setPlaceholderText("Em branco = usar o PATH do sistema")
        self._log_level_combo = QComboBox(tab)
        self._log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])

        form.addRow(
            pt("Diretorio de saida:"),
            self._with_help(
                self._output_directory_input,
                pt("Diretorio de saida"),
                self._help_html(
                    definition=(
                        "Nome da pasta derivada criada dentro do diretorio examinado para armazenar artefatos "
                        "de execucao, inventarios, recortes, quadros anotados, relatorio e registros."
                    ),
                    operational_effect=(
                        "Esse campo define onde os artefatos derivados ficam segregados dos arquivos de origem, "
                        "o que favorece rastreabilidade, reproducibilidade e preservacao dos originais."
                    ),
                    recommendation=(
                        "Use um nome fixo, curto e institucionalmente padronizado, como "
                        "'inventario_faces_output'. Evite reutilizar nomes aleatorios entre casos, porque isso "
                        "dificulta auditoria e automacao de limpeza controlada."
                    ),
                    caveat=(
                        "O aplicativo nao altera os originais, mas a separacao entre acervo e derivados deve "
                        "permanecer evidente em toda a documentacao tecnica."
                    ),
                    references=[
                        ("NIST - chain of custody", NIST_CHAIN_OF_CUSTODY_URL),
                        ("NIST CFReDS", NIST_CFREDS_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Titulo do relatorio:"),
            self._with_help(
                self._report_title_input,
                pt("Titulo do relatorio"),
                self._help_html(
                    definition=(
                        "Titulo formal impresso na capa e nos cabecalhos do relatorio tecnico emitido ao final "
                        "do processamento."
                    ),
                    operational_effect=(
                        "O titulo ajuda a identificar a natureza do exame, a diferenciar versoes do documento "
                        "e a manter coerencia entre o artefato documental e o escopo pericial."
                    ),
                    recommendation=(
                        "Use um titulo objetivo, por exemplo 'Relatorio Forense de Inventario Facial'. "
                        "Evite nomes promocionais ou conclusivos; a terminologia deve refletir que o resultado "
                        "tem natureza assistiva e probabilistica."
                    ),
                    caveat=(
                        "O titulo nao deve sugerir identificacao conclusiva de pessoas. Prefira linguagem "
                        "descritiva e tecnicamente neutra."
                    ),
                    references=[
                        ("NIST FRVT 1:N API", NIST_FRVT_URL),
                        ("InsightFace (repositorio oficial)", INSIGHTFACE_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Organizacao responsavel:"),
            self._with_help(
                self._organization_input,
                pt("Organizacao responsavel"),
                self._help_html(
                    definition=(
                        "Identificacao da unidade, laboratorio, delegacia, setor ou organizacao responsavel "
                        "pela emissao do relatorio."
                    ),
                    operational_effect=(
                        "Esse dado integra a cadeia documental do exame e facilita rastrear autoria institucional, "
                        "procedimentos adotados e origem administrativa do produto pericial."
                    ),
                    recommendation=(
                        "Informe a denominacao oficial e completa da unidade emissora. Evite siglas internas "
                        "sem contexto quando o documento puder circular fora do setor de origem."
                    ),
                    references=[
                        ("NIST - chain of custody", NIST_CHAIN_OF_CUSTODY_URL),
                        ("NIST CFReDS", NIST_CFREDS_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Nivel de log:"),
            self._with_help(
                self._log_level_combo,
                pt("Nivel de log"),
                self._help_html(
                    definition=(
                        "Controla a quantidade de detalhes registrada nos arquivos de log textuais e nos eventos "
                        "estruturados da execucao."
                    ),
                    operational_effect=(
                        "Niveis mais verbosos ampliam a capacidade de auditoria e diagnostico, mas tambem aumentam "
                        "o volume de dados persistidos. Em acervos grandes, isso impacta tamanho dos logs e tempo "
                        "de revisao humana."
                    ),
                    recommendation=(
                        "Use INFO na operacao de rotina. Use DEBUG para calibracao, homologacao e investigacao de "
                        "falhas. Reserve WARNING e ERROR para fluxos muito controlados, quando a equipe ja conhece "
                        "o comportamento esperado do ambiente."
                    ),
                    references=[
                        ("NIST CFReDS", NIST_CFREDS_URL),
                        ("NIST - chain of custody", NIST_CHAIN_OF_CUSTODY_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            "Diretório do MediaInfo:",
            self._with_help(
                self._with_directory_selector(
                    self._mediainfo_directory_input,
                    "Selecionar diretório do MediaInfo",
                ),
                "Diretório do MediaInfo",
                self._help_html(
                    definition=(
                        "Diretório que contém o executável do MediaInfo, normalmente nomeado como "
                        "'mediainfo.exe'. Quando preenchido, o aplicativo procura primeiro nessa pasta "
                        "antes de tentar localizar o utilitário pelo PATH do sistema."
                    ),
                    operational_effect=(
                        "Essa configuração elimina dependência exclusiva do PATH e evita falso negativo em "
                        "estações nas quais o MediaInfo está instalado em diretório próprio, portátil ou "
                        "fora das variáveis globais do Windows."
                    ),
                    recommendation=(
                        "Aponte para a pasta exata onde está o executável do MediaInfo. Deixe em branco "
                        "somente quando o utilitário já estiver disponível no PATH e isso tiver sido "
                        "validado na própria estação de trabalho."
                    ),
                    caveat=(
                        "O campo espera a pasta do executável. Se o diretório estiver incorreto, a coleta "
                        "de características técnicas de imagem e vídeo será marcada como indisponível no "
                        "inventário e no relatório."
                    ),
                    references=[
                        ("MediaInfo (site oficial)", MEDIAINFO_URL),
                        ("NIST CFReDS", NIST_CFREDS_URL),
                    ],
                ),
            ),
        )
        return tab

    def _build_media_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._image_extensions_input = QLineEdit(tab)
        self._image_extensions_input.setPlaceholderText(".jpg, .jpeg, .png")
        self._video_extensions_input = QLineEdit(tab)
        self._video_extensions_input.setPlaceholderText(".mp4, .avi, .mkv")

        self._sampling_interval_spin = QDoubleSpinBox(tab)
        self._sampling_interval_spin.setRange(0.1, 3600.0)
        self._sampling_interval_spin.setDecimals(2)
        self._sampling_interval_spin.setSingleStep(0.1)
        self._sampling_interval_spin.setSuffix(" s")

        self._max_frames_spin = QSpinBox(tab)
        self._max_frames_spin.setRange(1, 1_000_000)

        self._unlimited_frames_checkbox = QCheckBox(pt("Sem limite de quadros por video"), tab)
        self._unlimited_frames_checkbox.toggled.connect(self._max_frames_spin.setDisabled)

        max_frames_layout = QVBoxLayout()
        max_frames_layout.addWidget(self._max_frames_spin)
        max_frames_layout.addWidget(self._unlimited_frames_checkbox)
        max_frames_widget = QWidget(tab)
        max_frames_widget.setLayout(max_frames_layout)

        form.addRow(
            pt("Extensoes de imagem:"),
            self._with_help(
                self._image_extensions_input,
                pt("Extensoes de imagem"),
                self._help_html(
                    definition=(
                        "Lista de extensoes classificadas como imagem estatica pelo mecanismo de varredura "
                        "do aplicativo."
                    ),
                    operational_effect=(
                        "A classificacao determina quais arquivos seguem para leitura por rotinas de imagem. "
                        "Incluir extensoes indevidas aumenta falhas de abertura e ruido operacional; excluir "
                        "extensoes relevantes pode omitir evidencias visuais."
                    ),
                    recommendation=(
                        "Inclua apenas formatos efetivamente presentes no acervo e suportados no ambiente. "
                        "Em geral, mantenha .jpg, .jpeg e .png, e acrescente outros formatos somente apos "
                        "teste controlado."
                    ),
                    references=[
                        ("OpenCV imgcodecs: imread / imdecode", OPENCV_IMGCODECS_URL),
                        ("NIST CFReDS", NIST_CFREDS_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Extensoes de video:"),
            self._with_help(
                self._video_extensions_input,
                pt("Extensoes de video"),
                self._help_html(
                    definition=(
                        "Lista de extensoes tratadas como arquivo de video para fins de abertura, amostragem "
                        "temporal e extracao de quadros."
                    ),
                    operational_effect=(
                        "Essa lista controla o escopo do pipeline de video. Se estiver muito ampla, o sistema "
                        "tentara abrir arquivos nao decodificaveis. Se estiver muito restrita, arquivos "
                        "pertinentes poderao ficar fora da analise facial."
                    ),
                    recommendation=(
                        "Mantenha apenas formatos validados no ambiente de producao, como .mp4, .avi e .mkv. "
                        "Sempre teste novos formatos com um conjunto pequeno antes de usa-los em lote."
                    ),
                    references=[
                        ("OpenCV VideoCapture", OPENCV_VIDEOCAPTURE_URL),
                        ("NIST CFReDS", NIST_CFREDS_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Intervalo de amostragem:"),
            self._with_help(
                self._sampling_interval_spin,
                pt("Intervalo de amostragem"),
                self._help_html(
                    definition=(
                        "Espacamento temporal, em segundos, entre os quadros extraidos de um video para "
                        "analise facial."
                    ),
                    operational_effect=(
                        "Intervalos menores aumentam a cobertura temporal e a chance de capturar variacao facial, "
                        "mas elevam custo computacional, volume de artefatos e tempo total de processamento. "
                        "Intervalos maiores reduzem custo, porem podem perder aparicoes breves."
                    ),
                    recommendation=(
                        "Comece com 1,0 s a 2,0 s em triagem geral. Reduza o intervalo quando o caso exigir "
                        "maior granularidade temporal ou quando os rostos aparecem por poucos quadros."
                    ),
                    references=[
                        ("OpenCV VideoCapture", OPENCV_VIDEOCAPTURE_URL),
                        ("NIST FRVT 1:N API", NIST_FRVT_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Maximo de quadros por video:"),
            self._with_help(
                max_frames_widget,
                pt("Maximo de quadros por video"),
                self._help_html(
                    definition=(
                        "Limite superior de quadros amostrados por arquivo de video. Quando desativado, o "
                        "sistema percorre todo o arquivo conforme o intervalo de amostragem definido."
                    ),
                    operational_effect=(
                        "Esse limite e um mecanismo de controle de custo para videos extensos. Ele ajuda a "
                        "conter tempo de execucao, uso de armazenamento e volume de galerias e logs em acervos "
                        "muito grandes."
                    ),
                    recommendation=(
                        "Mantenha limite ativo em analises exploratorias ou acervos volumosos. Desative apenas "
                        "quando houver justificativa tecnica para cobertura integral e infraestrutura suficiente "
                        "para sustentar o aumento de processamento."
                    ),
                    caveat=(
                        "Ao impor limite, registre essa decisao metodologica no contexto do exame para que a "
                        "cobertura temporal parcial fique explicitada."
                    ),
                    references=[
                        ("OpenCV VideoCapture", OPENCV_VIDEOCAPTURE_URL),
                        ("NIST CFReDS", NIST_CFREDS_URL),
                    ],
                ),
            ),
        )
        return tab

    def _build_face_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._backend_input = QLineEdit(tab)
        self._model_name_input = QLineEdit(tab)
        self._ctx_id_spin = QSpinBox(tab)
        self._ctx_id_spin.setRange(-1, 64)

        self._det_width_spin = QSpinBox(tab)
        self._det_width_spin.setRange(32, 4096)
        self._det_width_spin.setSingleStep(32)
        self._det_height_spin = QSpinBox(tab)
        self._det_height_spin.setRange(32, 4096)
        self._det_height_spin.setSingleStep(32)

        det_size_layout = QHBoxLayout()
        det_size_layout.addWidget(self._det_width_spin)
        det_size_layout.addWidget(QLabel("x", tab))
        det_size_layout.addWidget(self._det_height_spin)
        self._original_resolution_checkbox = QCheckBox(pt("Manter resolução original do arquivo ou quadro"), tab)
        self._original_resolution_checkbox.toggled.connect(self._det_width_spin.setDisabled)
        self._original_resolution_checkbox.toggled.connect(self._det_height_spin.setDisabled)
        det_size_widget = QWidget(tab)
        det_size_container = QVBoxLayout(det_size_widget)
        det_size_container.setContentsMargins(0, 0, 0, 0)
        det_size_container.addLayout(det_size_layout)
        det_size_container.addWidget(self._original_resolution_checkbox)

        self._minimum_face_quality_spin = QDoubleSpinBox(tab)
        self._minimum_face_quality_spin.setRange(0.0, 1.0)
        self._minimum_face_quality_spin.setDecimals(3)
        self._minimum_face_quality_spin.setSingleStep(0.01)

        self._minimum_face_size_spin = QSpinBox(tab)
        self._minimum_face_size_spin.setRange(1, 4096)
        self._minimum_face_size_spin.setSingleStep(4)
        self._minimum_face_size_spin.setSuffix(" px")

        self._providers_input = QLineEdit(tab)
        self._providers_input.setPlaceholderText(pt("Em branco = selecao automatica"))

        form.addRow(
            pt("Mecanismo:"),
            self._with_help(
                self._backend_input,
                pt("Mecanismo"),
                self._help_html(
                    definition=(
                        "Implementacao de analise facial utilizada pelo aplicativo para detectar faces e "
                        "extrair vetores de caracteristicas."
                    ),
                    operational_effect=(
                        "O mecanismo define compatibilidade com modelos, comportamento de deteccao, formato "
                        "dos embeddings, dependencia de hardware e maturidade do ecossistema operacional."
                    ),
                    recommendation=(
                        "Mantenha 'insightface' quando a estacao estiver homologada para esse backend. "
                        "So substitua por outro mecanismo quando houver validacao tecnica e documental."
                    ),
                    references=[
                        ("InsightFace (repositorio oficial)", INSIGHTFACE_URL),
                        ("Deng et al. - ArcFace (CVPR 2019)", ARCFACE_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Modelo:"),
            self._with_help(
                self._model_name_input,
                pt("Modelo"),
                self._help_html(
                    definition=(
                        "Nome do conjunto de modelos carregado pelo backend facial. No ecossistema InsightFace, "
                        "isso normalmente combina detector, extrator de embeddings e componentes auxiliares."
                    ),
                    operational_effect=(
                        "O modelo influencia robustez, velocidade, consumo de memoria e capacidade de lidar com "
                        "variacoes de pose, iluminacao, resolucao e qualidade de captura."
                    ),
                    recommendation=(
                        "Use 'buffalo_l' como ponto de partida em estacoes com recursos compativeis, porque ele "
                        "costuma oferecer bom equilibrio entre robustez e maturidade. Em hardware restrito, valide "
                        "alternativas menores antes de usa-las em producao."
                    ),
                    references=[
                        ("InsightFace (repositorio oficial)", INSIGHTFACE_URL),
                        ("Deng et al. - ArcFace (CVPR 2019)", ARCFACE_URL),
                        ("Guo et al. - SCRFD", SCRFD_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Contexto:"),
            self._with_help(
                self._ctx_id_spin,
                pt("Contexto"),
                self._help_html(
                    definition=(
                        "Identificador do dispositivo de execucao usado pelo backend facial. Em geral, 0 indica "
                        "o primeiro dispositivo acelerado disponivel e -1 forca execucao em CPU."
                    ),
                    operational_effect=(
                        "Esse campo define onde o grafo sera executado, afetando desempenho, disponibilidade de "
                        "memoria, estabilidade do ambiente e reprodutibilidade operacional."
                    ),
                    recommendation=(
                        "Use 0 quando houver aceleracao previamente homologada. Use -1 para CPU quando a estacao "
                        "nao tiver suporte consistente a GPU, quando houver falhas de driver ou quando a prioridade "
                        "for estabilidade reprodutivel."
                    ),
                    references=[
                        ("ONNX Runtime Execution Providers", ONNXRUNTIME_EP_URL),
                        ("InsightFace (repositorio oficial)", INSIGHTFACE_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Tamanho de deteccao:"),
            self._with_help(
                det_size_widget,
                pt("Tamanho de deteccao"),
                self._help_html(
                    definition=(
                        "Resolucao de entrada usada pelo detector facial antes da inferencia. Ela nao altera o "
                        "arquivo original; apenas define a escala operacional do detector."
                    ),
                    operational_effect=(
                        "Valores maiores tendem a melhorar a resposta para faces pequenas, mas aumentam tempo de "
                        "inferencia e uso de memoria. Valores menores reduzem custo computacional, porem podem "
                        "perder deteccoes em rostos pequenos ou distantes."
                    ),
                    recommendation=(
                        "Use 640x640 como configuracao inicial para triagem. Suba para 800x800 ou 1024x1024 "
                        "somente quando houver evidencia de perda de rostos pequenos e a infraestrutura suportar "
                        "o aumento de custo."
                    ),
                    caveat=(
                        "Quando a opcao de resolucao original estiver marcada, o detector sera reajustado para a "
                        "largura e a altura do proprio arquivo ou quadro analisado. Isso amplia fidelidade "
                        "geometrica, mas pode elevar tempo de inferencia e consumo de memoria em arquivos grandes."
                    ),
                    references=[
                        ("Guo et al. - SCRFD", SCRFD_URL),
                        ("InsightFace (repositorio oficial)", INSIGHTFACE_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Qualidade minima da face:"),
            self._with_help(
                self._minimum_face_quality_spin,
                pt("Qualidade minima da face"),
                self._help_html(
                    definition=(
                        "Limiar minimo da pontuacao de deteccao para que uma face siga para o inventario. "
                        "No aplicativo, esse valor atua como filtro de aceitacao da ocorrencia."
                    ),
                    operational_effect=(
                        "Aumentar o limiar tende a reduzir falsos positivos e ruido visual, mas pode eliminar "
                        "rostos reais capturados em baixa qualidade. Reduzir demais aumenta sensibilidade, mas "
                        "eleva a chance de deteccoes espurias."
                    ),
                    recommendation=(
                        "Comece em 0,60 e calibre com amostras do proprio caso. Suba o valor quando o acervo "
                        "gerar muitas deteccoes ruins; reduza apenas com justificativa tecnica e registro dessa "
                        "decisao metodologica."
                    ),
                    caveat=(
                        "Nao existe um valor universalmente correto. A calibracao e dependente do caso, do sensor, "
                        "da distancia, da iluminacao e do objetivo operacional. O resultado permanece probabilistico "
                        "e investigativo."
                    ),
                    references=[
                        ("InsightFace (repositorio oficial)", INSIGHTFACE_URL),
                        ("Deng et al. - ArcFace (CVPR 2019)", ARCFACE_URL),
                        ("NIST FRVT 1:N API", NIST_FRVT_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Tamanho minimo da face:"),
            self._with_help(
                self._minimum_face_size_spin,
                pt("Tamanho minimo da face"),
                self._help_html(
                    definition=(
                        "Menor dimensao aceitavel, em pixels, para a caixa delimitadora da face. O filtro usa o "
                        "menor lado do retangulo detectado para decidir se a ocorrencia entra no inventario."
                    ),
                    operational_effect=(
                        "Faces muito pequenas costumam produzir recortes pouco informativos, elevar a taxa de "
                        "ruido visual e degradar a utilidade do agrupamento. Ao mesmo tempo, um valor muito alto "
                        "pode descartar rostos distantes que ainda teriam valor investigativo."
                    ),
                    recommendation=(
                        "Use 40 px a 64 px como faixa inicial em triagem geral. Reduza esse valor somente quando "
                        "o caso envolver capturas distantes e houver aceitacao consciente de maior incerteza. "
                        "Aumente quando o objetivo for privilegio de qualidade sobre abrangencia."
                    ),
                    caveat=(
                        "Esta recomendacao e operacional, nao normativa. O valor ideal deve ser calibrado com "
                        "material representativo do acervo real."
                    ),
                    references=[
                        ("Guo et al. - SCRFD", SCRFD_URL),
                        ("InsightFace (repositorio oficial)", INSIGHTFACE_URL),
                        ("NIST FRVT 1:N API", NIST_FRVT_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Mecanismos de execucao:"),
            self._with_help(
                self._providers_input,
                pt("Mecanismos de execucao"),
                self._help_html(
                    definition=(
                        "Lista, em ordem de preferencia, dos mecanismos de execucao do ONNX Runtime que o "
                        "aplicativo tentara usar para inferencia."
                    ),
                    operational_effect=(
                        "A ordem dos mecanismos interfere diretamente em desempenho, compatibilidade e estabilidade. "
                        "Uma configuracao inadequada pode levar a queda para CPU, falhas de inicializacao ou "
                        "resultados inconsistentes entre estacoes."
                    ),
                    recommendation=(
                        "Deixe em branco para selecao automatica quando o ambiente ainda estiver em homologacao. "
                        "Informe explicitamente os mecanismos apenas quando a infraestrutura ja estiver validada, "
                        "por exemplo com CUDAExecutionProvider, DmlExecutionProvider e CPUExecutionProvider."
                    ),
                    references=[
                        ("ONNX Runtime Execution Providers", ONNXRUNTIME_EP_URL),
                        ("InsightFace (repositorio oficial)", INSIGHTFACE_URL),
                    ],
                ),
            ),
        )
        return tab

    def _build_clustering_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._assignment_similarity_spin = QDoubleSpinBox(tab)
        self._assignment_similarity_spin.setRange(0.0, 1.0)
        self._assignment_similarity_spin.setDecimals(3)
        self._assignment_similarity_spin.setSingleStep(0.01)

        self._candidate_similarity_spin = QDoubleSpinBox(tab)
        self._candidate_similarity_spin.setRange(0.0, 1.0)
        self._candidate_similarity_spin.setDecimals(3)
        self._candidate_similarity_spin.setSingleStep(0.01)

        self._min_cluster_size_spin = QSpinBox(tab)
        self._min_cluster_size_spin.setRange(1, 10_000)

        form.addRow(
            pt("Limiar de atribuicao:"),
            self._with_help(
                self._assignment_similarity_spin,
                pt("Limiar de atribuicao"),
                self._help_html(
                    definition=(
                        "Similaridade minima por cosseno exigida para que uma nova ocorrencia seja incorporada "
                        "a um possivel individuo ja existente."
                    ),
                    operational_effect=(
                        "Valores altos reduzem o risco de fusao indevida entre individuos diferentes, mas tendem "
                        "a fragmentar o resultado em mais grupos. Valores baixos aumentam cobertura, porem elevam "
                        "o risco de juntar rostos distintos."
                    ),
                    recommendation=(
                        "Comece entre 0,50 e 0,60 e valide com amostras do proprio caso. Ajuste o limiar apenas "
                        "depois de revisar visualmente as galerias e as ocorrencias representativas."
                    ),
                    caveat=(
                        "Esse limiar nao equivale a identificacao positiva. Ele serve apenas para agrupamento "
                        "assistido e requer revisao humana."
                    ),
                    references=[
                        ("Deng et al. - ArcFace (CVPR 2019)", ARCFACE_URL),
                        ("NIST FRVT 1:N API", NIST_FRVT_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Limiar de sugestao entre grupos:"),
            self._with_help(
                self._candidate_similarity_spin,
                pt("Limiar de sugestao entre grupos"),
                self._help_html(
                    definition=(
                        "Similaridade por cosseno usada para sinalizar que dois possiveis individuos distintos "
                        "merecem revisao como possivel correlacao."
                    ),
                    operational_effect=(
                        "Esse valor produz alertas investigativos adicionais. Se estiver muito alto, poucas "
                        "correlacoes serao sugeridas. Se estiver muito baixo, o numero de alertas cresce e pode "
                        "sobrecarregar a revisao humana."
                    ),
                    recommendation=(
                        "Mantenha esse limiar abaixo do limiar de atribuicao, para que ele funcione como zona "
                        "de atencao e nao como fusao automatica. Em geral, 0,40 a 0,50 e um intervalo inicial util."
                    ),
                    caveat=(
                        "Trate as sugestoes como indicio de revisao, nunca como conclusao de identidade."
                    ),
                    references=[
                        ("Deng et al. - ArcFace (CVPR 2019)", ARCFACE_URL),
                        ("NIST FRVT 1:N API", NIST_FRVT_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Tamanho minimo do grupo:"),
            self._with_help(
                self._min_cluster_size_spin,
                pt("Tamanho minimo do grupo"),
                self._help_html(
                    definition=(
                        "Quantidade minima de ocorrencias exigida para que um possivel individuo permaneca "
                        "na saida final do agrupamento."
                    ),
                    operational_effect=(
                        "Elevar esse valor reduz ruido visual e grupos muito pequenos, mas tambem pode remover "
                        "individuos raros, aparicoes unicas e evidencias pontuais. Valores baixos preservam mais "
                        "cobertura, ao custo de maior revisao manual."
                    ),
                    recommendation=(
                        "Use 1 quando a prioridade for completude do inventario. Suba para 2 ou 3 somente quando "
                        "houver necessidade de reduzir ruido em acervos grandes e quando a perda de ocorrencias "
                        "isoladas for aceitavel para o objetivo do exame."
                    ),
                    references=[
                        ("Deng et al. - ArcFace (CVPR 2019)", ARCFACE_URL),
                        ("NIST FRVT 1:N API", NIST_FRVT_URL),
                    ],
                ),
            ),
        )
        return tab

    def _build_reporting_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)

        self._max_gallery_faces_spin = QSpinBox(tab)
        self._max_gallery_faces_spin.setRange(1, 200)

        self._compile_pdf_checkbox = QCheckBox("Compilar PDF automaticamente", tab)

        self._chain_note_input = QPlainTextEdit(tab)
        self._chain_note_input.setPlaceholderText(pt("Informe a nota de cadeia de custodia."))
        self._chain_note_input.setMinimumHeight(150)

        form.addRow(
            pt("Faces maximas por individuo na galeria:"),
            self._with_help(
                self._max_gallery_faces_spin,
                pt("Faces maximas por individuo na galeria"),
                self._help_html(
                    definition=(
                        "Quantidade maxima de ocorrencias ilustradas para cada possivel individuo no relatorio."
                    ),
                    operational_effect=(
                        "Valores baixos favorecem leitura rapida e PDFs compactos. Valores altos aumentam a "
                        "representatividade visual do grupo, mas podem alongar o documento, ampliar o consumo "
                        "de disco e prejudicar legibilidade em casos extensos."
                    ),
                    recommendation=(
                        "Use entre 4 e 8 em operacao rotineira. Amplie esse limite somente quando houver "
                        "necessidade concreta de documentar maior variedade intra-grupo."
                    ),
                    references=[
                        ("NIST CFReDS", NIST_CFREDS_URL),
                        ("NIST FRVT 1:N API", NIST_FRVT_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Compilacao do PDF:"),
            self._with_help(
                self._compile_pdf_checkbox,
                pt("Compilacao do PDF"),
                self._help_html(
                    definition=(
                        "Define se o arquivo TEX sera convertido automaticamente para PDF ao final do "
                        "processamento."
                    ),
                    operational_effect=(
                        "Quando ativo, o fluxo entrega o documento final pronto para revisao. Quando desativado, "
                        "a execucao termina mais cedo e deixa apenas o TEX para compilacao posterior."
                    ),
                    recommendation=(
                        "Mantenha ativo em estacoes com LaTeX validado. Desative apenas durante calibracao, "
                        "depuracao ou quando a equipe quiser revisar o TEX antes da compilacao final."
                    ),
                    references=[
                        ("NIST CFReDS", NIST_CFREDS_URL),
                        ("NIST - chain of custody", NIST_CHAIN_OF_CUSTODY_URL),
                    ],
                ),
            ),
        )
        form.addRow(
            pt("Nota de cadeia de custodia:"),
            self._with_help(
                self._chain_note_input,
                pt("Nota de cadeia de custodia"),
                self._help_html(
                    definition=(
                        "Texto institucional inserido no relatorio para registrar a preservacao dos originais, "
                        "a segregacao dos derivados e a natureza assistiva do processamento."
                    ),
                    operational_effect=(
                        "Esse texto ajuda a explicitar o tratamento dado aos vestigios digitais e a situar o "
                        "relatorio dentro do procedimento documental do orgao."
                    ),
                    recommendation=(
                        "Padronize a redacao segundo o procedimento oficial da instituicao. Inclua, de forma "
                        "clara, que os arquivos originais nao sao alterados e que os resultados possuem carater "
                        "probabilistico e dependem de revisao humana."
                    ),
                    references=[
                        ("NIST - chain of custody", NIST_CHAIN_OF_CUSTODY_URL),
                        ("NIST CFReDS", NIST_CFREDS_URL),
                    ],
                ),
            ),
        )
        return tab

    def _load_config(self, config: AppConfig) -> None:
        self._output_directory_input.setText(config.app.output_directory_name)
        self._report_title_input.setText(config.app.report_title)
        self._organization_input.setText(config.app.organization)
        self._mediainfo_directory_input.setText(config.app.mediainfo_directory or "")
        self._log_level_combo.setCurrentText(config.app.log_level.upper())

        self._image_extensions_input.setText(", ".join(config.media.image_extensions))
        self._video_extensions_input.setText(", ".join(config.media.video_extensions))
        self._sampling_interval_spin.setValue(config.video.sampling_interval_seconds)
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

        self._assignment_similarity_spin.setValue(config.clustering.assignment_similarity)
        self._candidate_similarity_spin.setValue(config.clustering.candidate_similarity)
        self._min_cluster_size_spin.setValue(config.clustering.min_cluster_size)

        self._max_gallery_faces_spin.setValue(config.reporting.max_gallery_faces_per_group)
        self._compile_pdf_checkbox.setChecked(config.reporting.compile_pdf)
        self._chain_note_input.setPlainText(config.forensics.chain_of_custody_note)

    def _restore_selected_config(self) -> None:
        self._load_config(self._selected_config)

    def _accept_configuration(self) -> None:
        try:
            self._selected_config = self._build_config()
        except ValueError as exc:
            QMessageBox.warning(self, pt("Configuracao invalida"), pt(str(exc)))
            return
        self.accept()

    def _build_config(self) -> AppConfig:
        output_directory_name = self._require_text(
            self._output_directory_input.text(),
            pt("Informe o diretorio de saida da execucao."),
        )
        report_title = self._require_text(
            self._report_title_input.text(),
            pt("Informe o titulo do relatorio."),
        )
        organization = self._require_text(
            self._organization_input.text(),
            pt("Informe a organizacao responsavel."),
        )
        mediainfo_directory = self._optional_directory(
            self._mediainfo_directory_input.text(),
            "MediaInfo",
        )
        backend = self._require_text(self._backend_input.text(), pt("Informe o mecanismo facial."))
        model_name = self._require_text(self._model_name_input.text(), pt("Informe o nome do modelo facial."))
        chain_note = self._require_text(
            self._chain_note_input.toPlainText(),
            pt("Informe a nota de cadeia de custodia."),
        )

        return AppConfig(
            app=AppSettings(
                name=self._selected_config.app.name,
                output_directory_name=output_directory_name,
                report_title=report_title,
                organization=organization,
                log_level=self._log_level_combo.currentText(),
                mediainfo_directory=mediainfo_directory,
            ),
            media=MediaSettings(
                image_extensions=self._parse_extensions(self._image_extensions_input.text(), "imagem"),
                video_extensions=self._parse_extensions(self._video_extensions_input.text(), "video"),
            ),
            video=VideoSettings(
                sampling_interval_seconds=float(self._sampling_interval_spin.value()),
                max_frames_per_video=(
                    None if self._unlimited_frames_checkbox.isChecked() else int(self._max_frames_spin.value())
                ),
            ),
            face_model=FaceModelSettings(
                backend=backend,
                model_name=model_name,
                det_size=(
                    None
                    if self._original_resolution_checkbox.isChecked()
                    else (int(self._det_width_spin.value()), int(self._det_height_spin.value()))
                ),
                minimum_face_quality=float(self._minimum_face_quality_spin.value()),
                minimum_face_size_pixels=int(self._minimum_face_size_spin.value()),
                ctx_id=int(self._ctx_id_spin.value()),
                providers=self._parse_list(self._providers_input.text()),
            ),
            clustering=ClusteringSettings(
                assignment_similarity=float(self._assignment_similarity_spin.value()),
                candidate_similarity=float(self._candidate_similarity_spin.value()),
                min_cluster_size=int(self._min_cluster_size_spin.value()),
            ),
            reporting=ReportingSettings(
                max_gallery_faces_per_group=int(self._max_gallery_faces_spin.value()),
                compile_pdf=self._compile_pdf_checkbox.isChecked(),
            ),
            forensics=ForensicsSettings(chain_of_custody_note=chain_note),
        )

    def _with_help(self, widget: QWidget, title: str, body: str) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)

        help_button = QToolButton(container)
        help_button.setText("?")
        help_button.setToolTip(pt("Ajuda tecnica"))
        help_button.setFocusPolicy(Qt.NoFocus)
        help_button.clicked.connect(lambda _checked=False, t=pt(title), b=body: self._show_help(t, b))
        layout.addWidget(help_button)
        return container

    def _with_directory_selector(self, line_edit: QLineEdit, caption: str) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)

        browse_button = QPushButton("Selecionar...", container)
        browse_button.clicked.connect(
            lambda _checked=False, target=line_edit, dialog_caption=caption: self._select_directory(
                target,
                dialog_caption,
            )
        )
        layout.addWidget(browse_button)
        return container

    def _select_directory(self, target: QLineEdit, caption: str) -> None:
        initial_directory = target.text().strip() or ""
        selected_directory = QFileDialog.getExistingDirectory(self, caption, initial_directory)
        if selected_directory:
            target.setText(selected_directory)

    def _show_help(self, title: str, body: str) -> None:
        self._help_title_label.setText(pt(title))
        self._help_body.setHtml(body)

    def _help_html(
        self,
        *,
        definition: str,
        operational_effect: str,
        recommendation: str,
        references: list[tuple[str, str]],
        caveat: str | None = None,
    ) -> str:
        references_html = "".join(
            f"<li>{self._abnt_reference_html(label, url)}</li>" for label, url in references
        )
        caveat_html = f"<p><b>{pt('Observacao.')}</b> {pt(caveat)}</p>" if caveat else ""
        return (
            "<div style='font-family: Segoe UI, sans-serif; font-size: 10pt;'>"
            f"<p><b>{pt('Definicao.')}</b> {pt(definition)}</p>"
            f"<p><b>{pt('Efeito operacional.')}</b> {pt(operational_effect)}</p>"
            f"<p><b>{pt('Recomendacao tecnica.')}</b> {pt(recommendation)}</p>"
            f"{caveat_html}"
            f"<p><b>{pt('Referencias.')}</b></p>"
            f"<ul>{references_html}</ul>"
            "</div>"
        )

    def _abnt_reference_html(self, label: str, url: str) -> str:
        if url == MEDIAINFO_URL:
            return (
                "MEDIAAREA. <i>MediaInfo</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            )
        entries = {
            INSIGHTFACE_URL: (
                "INSIGHTFACE. <i>InsightFace: an open source 2D and 3D deep face analysis library</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            ONNXRUNTIME_EP_URL: (
                "MICROSOFT. <i>ONNX Runtime: execution providers</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            OPENCV_VIDEOCAPTURE_URL: (
                "OPENCV. <i>OpenCV 4.x documentation: VideoCapture class reference</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            OPENCV_IMGCODECS_URL: (
                "OPENCV. <i>OpenCV 4.x documentation: imgcodecs module reference</i>. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            SCRFD_URL: (
                "GUO, Jia et al. <i>Sample and computation redistribution for efficient face detection</i>. "
                f"arXiv, 2021. Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            ARCFACE_URL: (
                "DENG, Jiankang et al. <i>ArcFace: additive angular margin loss for deep face recognition</i>. "
                "In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. "
                f"Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            NIST_FRVT_URL: (
                "NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY. <i>Face recognition vendor test (FRVT) "
                "1:N identification</i>. Disponível em: "
                f"&lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            NIST_CHAIN_OF_CUSTODY_URL: (
                "NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY. <i>Chain of custody</i>. Disponível em: "
                f"&lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
            NIST_CFREDS_URL: (
                "NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY. <i>Computer Forensic Reference Data Sets "
                "(CFReDS)</i>. Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}"
            ),
        }
        return entries.get(
            url,
            f"{pt(label)}. Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. {ABNT_ACCESS_DATE}",
        )

    def _parse_extensions(self, raw_value: str, label: str) -> tuple[str, ...]:
        extensions = []
        for item in self._parse_list(raw_value):
            normalized = item.lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            extensions.append(normalized)
        if not extensions:
            raise ValueError(pt(f"Informe ao menos uma extensao de {label}."))
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
