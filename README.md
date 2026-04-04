# Inventario Faces

Aplicativo desktop em Python para inventário forense assistido de faces em imagens e vídeos, com foco em auditabilidade, agrupamento probabilístico e relatórios técnicos.

Repositório oficial: [https://github.com/demusis/inventario_faces](https://github.com/demusis/inventario_faces)

Issues e atualizações: [https://github.com/demusis/inventario_faces/issues](https://github.com/demusis/inventario_faces/issues)

## O que o sistema faz

- varre recursivamente um diretório;
- calcula hash SHA-512 de todos os arquivos;
- processa imagens e vídeos sem alterar os originais;
- rastreia faces persistentes em vídeo por `FaceTrack`;
- seleciona `KeyFrame` por qualidade, intervalo e mudança significativa;
- representa cada track por embedding médio normalizado;
- agrupa possíveis indivíduos por track, não por frame isolado;
- cria índice vetorial para busca coarse-to-fine;
- gera inventário estruturado, logs e relatório técnico em `.tex`, `.pdf` e `.docx`.

## Avisos críticos

- O sistema não realiza identificação conclusiva de indivíduos.
- Toda correspondência facial deve ser tratada como probabilística.
- Os arquivos originais não são sobrescritos.
- Todo processamento relevante é registrado para auditoria.

## Arquitetura

O pipeline foi reorganizado em módulos coesos:

- `services/tracking_service.py`: associação temporal, encerramento de tracks e seleção de keyframes.
- `services/enhancement_service.py`: pré-processamento auditável para baixa iluminação e ruído.
- `services/quality_service.py`: métricas de nitidez, iluminação e frontalidade.
- `services/clustering_service.py`: agrupamento em nível de track.
- `services/search_service.py`: indexação vetorial com preferência por FAISS e fallback em NumPy.
- `reporting/`: geração dos relatórios LaTeX e DOCX.

## Entidades principais

- `FaceOccurrence`: detecção individual com vínculo para `track_id`, `keyframe_id`, qualidade e metadados de melhoria.
- `FaceTrack`: trilha temporal da mesma face ao longo de um vídeo ou ocorrência unitária em imagem.
- `KeyFrame`: frame representativo salvo para auditoria, visualização e embedding de referência.
- `EnhancementMetadata`: descreve qualquer pré-processamento aplicado antes da detecção.

## Saídas geradas

Cada execução cria uma pasta `inventario_faces_output/run_YYYYMMDD_HHMMSS/` com:

- `inventory/files.csv`
- `inventory/occurrences.csv`
- `inventory/tracks.csv`
- `inventory/keyframes.csv`
- `inventory/clusters.json`
- `inventory/media_info.json`
- `inventory/search.json`
- `inventory/manifest.json`
- `logs/run.log`
- `logs/events.jsonl`
- `artifacts/crops/`
- `artifacts/contexts/`
- `report/relatorio_forense.tex`
- `report/relatorio_forense.pdf`
- `report/relatorio_forense.docx`

## Configuração externa

Os parâmetros ficam em `config/defaults.yaml` e podem ser sobrescritos por configuração persistente do usuário. A interface agora expõe os grupos técnicos do pipeline, e os metadados de mídia são extraídos internamente pelo próprio aplicativo.

- `video`: taxa de amostragem, teto de quadros, intervalo de keyframe e limiar de mudança significativa.
- `face_model`: backend, modelo, resolução de detecção, qualidade mínima, tamanho mínimo e execution providers.
- `tracking`: IoU, distância espacial, similaridade de embedding, pesos de associação, tolerância a perda e top crops.
- `clustering`: limiares de atribuição/sugestão e mínimos de grupo e de track.
- `enhancement`: pré-processamento, brilho-gatilho, CLAHE, gamma e denoise.
- `search`: habilitação da indexação, preferência por FAISS, coarse search e refine search.
- `reporting`: compilação do PDF, densidade das seções e nota de cadeia de custódia.

Por padrão, o scanner já considera extensões usuais de vídeo como `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.mpeg`, `.mpg` e `.dav`. Para `.dav`, a leitura efetiva continua dependendo do suporte de codec disponível no ambiente OpenCV/FFmpeg da estação.

## Instalação

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Para análise facial real:

```powershell
pip install "insightface>=0.7.3"
```

Opcionalmente, para busca vetorial com FAISS:

```powershell
pip install "faiss-cpu>=1.8"
```

## Execução

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
.\.venv\Scripts\Activate.ps1
python -m inventario_faces
```

Fluxo esperado na interface:

1. selecionar a pasta de entrada;
2. revisar os parâmetros em `Configurações`;
3. clicar em `Criar inventário`;
4. abrir o relatório e a pasta da execução ao final.

## Exemplo de resultado esperado

Para um vídeo com a mesma pessoa aparecendo em vários frames:

- várias detecções geram um único `FaceTrack`;
- apenas os melhores frames viram `KeyFrame`;
- o clustering usa o embedding do track;
- o relatório exibe tabelas por grupo, quadro de origem com a área destacada, timeline textual e possíveis correlações entre grupos.

## Testes

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

A suíte cobre:

- tracking correto;
- redução de redundância por keyframe;
- clustering consistente;
- reprodutibilidade;
- persistência de configuração;
- geração de relatório.

## Requisitos

- Windows 10 ou superior
- Python 3.11 ou 3.12
- `pdflatex` disponível no sistema para compilação automática do PDF
- extração interna de metadados técnicos de imagem e vídeo via bibliotecas do próprio aplicativo

## Empacotamento

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
powershell -ExecutionPolicy Bypass -File .\build\build.ps1
```

O instalador Windows gerado pelo script aponta para o repositório oficial do projeto como canal de publicação, suporte e atualização.

## Observação final

O sistema foi desenhado para apoiar triagem e análise pericial, não para substituir juízo técnico humano.
