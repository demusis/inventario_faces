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
- executa `Busca por faces` com uma ou mais imagens de consulta, preservando no relatório também as consultas rejeitadas;
- compara conjuntos `Padrão` e `Questionado`, com ranking par-a-par, malha biométrica e revisão visual;
- calibra scores por razão de verossimilhança (LR) a partir de base rotulada ou de modelo JSON já salvo;
- gera inventário estruturado, logs e relatório técnico em `.tex`, `.pdf` e `.docx`.

## Avisos críticos

- O sistema não realiza identificação conclusiva de indivíduos.
- Toda correspondência facial deve ser tratada como probabilística.
- Os arquivos originais não são sobrescritos.
- Todo processamento relevante é registrado para auditoria.

## Arquitetura

O pipeline foi reorganizado em módulos coesos:

- `services/inventory_service.py`: orquestração do inventário (local e distribuído).
- `services/face_search_service.py`: operação completa de `Busca por faces`.
- `services/face_set_comparison_service.py`: comparação `Padrão` vs `Questionado`.
- `services/lr_calibration.py`: calibração e aplicação da razão de verossimilhança (LR).
- `services/tracking_service.py`: associação temporal, encerramento de tracks e seleção de keyframes.
- `services/enhancement_service.py`: pré-processamento auditável para baixa iluminação e ruído.
- `services/quality_service.py`: métricas de nitidez, iluminação e frontalidade.
- `services/clustering_service.py`: agrupamento em nível de track, com ordem determinística, centroide ponderado pelo tamanho do track e passe de refinamento.
- `services/search_service.py`: indexação vetorial com preferência por FAISS e fallback em NumPy.
- `services/pipeline_support.py` e `services/partial_payloads.py`: helpers compartilhados de log/progresso/IO atômico e desserialização dos parciais distribuídos.
- `infrastructure/sync_drive.py`: detecção de drives de sincronização (Google Drive/OneDrive/Dropbox) para alertar no modo distribuído.
- `gui/comparison_statistics.py` e `gui/comparison_widgets.py`: estatística explicativa e widgets de visualização da comparação.
- `reporting/`: geração dos relatórios LaTeX e DOCX.

Comportamentos de auditoria relevantes:

- no modo distribuído, se o diretório de evidências ou de execução compartilhada estiver em um drive de sincronização, o aplicativo emite alerta no log e na trilha de eventos (a execução não é bloqueada; prefira compartilhamento SMB/UNC);
- na calibração LR, pares com escore fora do intervalo de scores usado na calibração são sinalizados (`outside_calibration_support`) no JSON, no CSV, no resumo e no rótulo de evidência, pois o LR desses pares é extrapolado;
- nos relatórios, quando um grupo tem mais tracks do que o teto configurado, o truncamento é declarado explicitamente com referência ao `inventory/tracks.csv`.

## Entidades principais

- `FaceOccurrence`: detecção individual com vínculo para `track_id`, `keyframe_id`, qualidade e metadados de melhoria.
- `FaceTrack`: trilha temporal da mesma face ao longo de um vídeo ou ocorrência unitária em imagem.
- `KeyFrame`: frame representativo salvo para auditoria, visualização e embedding de referência.
- `EnhancementMetadata`: descreve qualquer pré-processamento aplicado antes da detecção.

## Saídas geradas

Cada execução cria uma pasta própria sob `inventario_faces_output/`. No inventário principal, o formato típico é `run_YYYYMMDD_HHMMSS/`, com:

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

Operações especializadas geram subpastas próprias dentro da execução:

- `search/`: artefatos estruturados da `Busca por faces`, incluindo relatório, eventos das consultas e compatibilidades encontradas;
- `comparison/`: artefatos da `Comparação entre grupos faciais`, incluindo ranking de pares, resumo estatístico, JSON da comparação e arquivos da calibração LR.

## Configuração externa

Os parâmetros ficam em `config/defaults.yaml` e podem ser sobrescritos por configuração persistente do usuário. A interface agora expõe os grupos técnicos do pipeline, e os metadados de mídia são extraídos internamente pelo próprio aplicativo.

Os valores padrão atuais foram endurecidos para um uso pericial mais conservador. Em especial, a configuração base passou a privilegiar:

- amostragem temporal mais densa em vídeo;
- ausência de teto padrão de quadros amostrados por vídeo;
- seleção automática de provider, com preferência por GPU quando disponível e fallback para CPU;
- thresholds mais conservadores para tracking e clustering;
- busca vetorial com maior cobertura de candidatos;
- calibração LR com estimador de densidade não paramétrico limitado ao suporte do score;
- nota institucional explícita sobre a natureza probabilística dos resultados.

- `media`: extensões de imagem (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`) e de vídeo (`.mp4`, `.avi`, `.mkv`, `.dav`, `.mov`, `.wmv`, `.mpeg`, `.mpg`) reconhecidas pelo scanner.
- `video`: taxa de amostragem, teto de quadros, intervalo de keyframe e limiar de mudança significativa.
- `comparison`: teto opcional de quadros por vídeo e intervalo de amostragem próprios da comparação `Padrão` x `Questionado` (nulos por padrão = herdam `video`); e os limiares de política do veredito por mídia quando há calibração LR — `verdict_assignment_log10_lr` (padrão 4.0, ou seja LR ≥ 10⁴, para atribuição) e `verdict_candidate_log10_lr` (padrão 1.0, LR ≥ 10, para candidata).
- `face_model`: backend, modelo, resolução de detecção (`det_size`), qualidade mínima, tamanho mínimo, ID de contexto do dispositivo (`ctx_id`) e execution providers. Por padrão, `providers` é vazio (descoberta automática CUDA → DirectML → CPU). Para ativar o `TensorrtExecutionProvider` (opt-in, mais rápido sobre CUDA, porém com warm-up longo de build de engine), liste-o explicitamente em `providers`.
- `tracking`: IoU, distância espacial, similaridade de embedding, pesos de associação, tolerância a perda e top crops.
- `clustering`: limiares de atribuição/sugestão e mínimos de grupo e de track.
- `enhancement`: pré-processamento, brilho-gatilho, CLAHE, gamma e denoise.
- `search`: habilitação da indexação, preferência por FAISS, coarse search e refine search.
- `likelihood_ratio`: amostragem máxima, mínimos de suporte, estimador de densidade e parâmetros de suavização da LR.
- `distributed`: modo compartilhado, identificador do lote, heartbeat do nó, timeout de lock órfão, auto-finalização, validação de integridade dos parciais e recuperação automática de itens ausentes/corrompidos.
- `app`: nome institucional, pasta derivada de saída, nível de log e uso opcional de cópia temporária local da mídia antes da leitura.
- `reporting`: compilação do PDF, densidade das seções e nota de cadeia de custódia.
- `forensics`: nota obrigatória de cadeia de custódia incluída em todos os relatórios.

Por padrão, o scanner já considera extensões usuais de vídeo como `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.mpeg`, `.mpg` e `.dav`. Para `.dav`, a leitura efetiva continua dependendo do suporte de codec disponível no ambiente OpenCV/FFmpeg da estação.

### Validação de configuração

O aplicativo agora valida a configuração de forma explícita antes de iniciar o processamento. Entre as verificações aplicadas estão:

- diretório de saída sem caminho absoluto ou relativo embutido;
- extensões de mídia normalizadas;
- limiares probabilísticos mantidos entre `0.0` e `1.0`;
- pesos, contagens e intervalos temporais com valores positivos;
- coerência entre limiares de clustering;
- coerência entre validação de parciais e recuperação automática no modo distribuído.

Valores booleanos persistidos em YAML também são interpretados de forma segura, aceitando `true/false`, `yes/no`, `on/off` e `1/0`. Quando a configuração estiver inválida, o erro informa o arquivo de origem para facilitar correção e auditoria.

Na interface, as alterações aplicadas em `Configurações` continuam sendo persistidas para as próximas execuções. O diálogo agora oferece dois restauros distintos:

- `Restaurar valores carregados`: volta ao estado que estava ativo ao abrir o diálogo;
- `Restaurar valores padrão`: volta ao baseline oficial do aplicativo. Ao aplicar, esse baseline também fica persistido como nova configuração ativa do usuário.

## Instalação

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

No Windows, `requirements.txt` já instala `onnxruntime-gpu[cuda,cudnn]`. O aplicativo tenta usar GPU automaticamente quando houver provider compatível e faz fallback para CPU quando necessário.

Ao processar vídeos, a decodificação de quadros (CPU) é sobreposta à inferência (GPU) por um prefetch determinístico: enquanto a GPU analisa o quadro atual, os próximos já vão sendo decodificados em segundo plano. A ordem dos quadros e os resultados são idênticos aos do processamento sequencial — apenas o tempo total diminui.

Para análise facial, o aplicativo embute dois backends equivalentes:

- `insightface` (padrão): usa o pacote InsightFace quando instalado;
- `onnx`: executa os mesmos modelos (`buffalo_l`) diretamente no ONNX Runtime, sem o pacote `insightface` — útil em estações com Python mais novo ou sem as dependências do InsightFace. Quando o pacote `insightface` está ausente ou quebrado, o aplicativo cai automaticamente nesse backend.

Os dois produzem embeddings numericamente equivalentes (mesmos arquivos de modelo). Para instalar o pacote InsightFace (opcional):

```powershell
pip install "insightface>=0.7.3"
```

Opcionalmente, para busca vetorial com FAISS:

```powershell
pip install "faiss-cpu>=1.8"
```

Importante: execute `python -m inventario_faces` com o mesmo interpretador em que o pacote foi instalado. Na prática, isso normalmente significa ativar a `.venv` do projeto antes de abrir o aplicativo.

## Execução

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
.\.venv\Scripts\Activate.ps1
python -m inventario_faces
```

Fluxo esperado na interface:

1. selecionar a pasta de entrada;
2. revisar os parâmetros em `Configurações`;
3. escolher entre `Criar inventário`, `Busca por faces` ou `Comparar conjuntos`;
4. abrir o relatório ou a pasta da execução ao final.

## Busca por faces

Na janela principal, `Busca por faces` aceita uma ou mais imagens de consulta.

- cada consulta passa pelo mesmo pipeline de detecção, filtros e embeddings do inventário;
- quando uma consulta é válida, o sistema seleciona uma face de referência para aquela imagem;
- a pesquisa vetorial é executada por consulta e depois consolidada pelo maior score por entidade;
- consultas inválidas, corrompidas ou sem face elegível continuam aparecendo no relatório para fins de cadeia de custódia.

## Comparação entre grupos faciais e LR

A ação `Comparar conjuntos` foi desenhada para o fluxo **1:N de identificação**: o grupo `Padrão` é o conjunto de **referência**, contendo um **único indivíduo** (normalmente várias imagens dele), e o grupo `Questionado` reúne as mídias onde se quer **localizar esse indivíduo**.

- o `Questionado` aceita **imagens e vídeos misturados** (inclusive `.dav`); vídeos passam pela mesma amostragem temporal, detecção, tracking e seleção de faces do inventário;
- o `Padrão` é consolidado como **referência de um único indivíduo**. Um agrupamento de **controle de qualidade** é aplicado apenas ao Padrão para detectar contaminação: se ele se dividir em mais de um agrupamento, o relatório emite um **alerta** de que pode haver imagem de outra pessoa ou faces de baixa qualidade na referência;
- o `Questionado` **não é agrupado por indivíduo** — não interessa quantas pessoas há nele, e sim em quais mídias a referência aparece. Cada face/mídia do Questionado é confrontada diretamente contra a referência;
- a comparação é **par-a-par** entre todas as faces elegíveis do Padrão e do Questionado, com ranking e faixas decisórias (`atribuição`/`candidata`/`abaixo do limiar`) e razão de verossimilhança (LR) opcional — esse núcleo evidencial é inalterado;
- o resumo e o `run.log` relacionam, por **mídia do Questionado**, quais atingiram a faixa de atribuição e quais ficaram somente como candidatas, respondendo objetivamente se (e onde) o indivíduo de referência aparece. **Quando há calibração LR, esse veredito por mídia é governado pela razão de verossimilhança** (limiares `comparison.verdict_*_log10_lr`); sem calibração, recai nas faixas de similaridade do agrupamento — a base usada fica registrada no resumo e no log;
- a janela de LR pode usar uma base rotulada com uma subpasta por identidade ou um modelo LR já salvo em JSON;
- quando a calibração é calculada na execução, o modelo é salvo automaticamente para reaproveitamento posterior.

Para limitar o tempo em vídeos longos, a seção `comparison` da configuração permite um teto opcional de quadros amostrados (`max_frames_per_video`) e/ou um intervalo de amostragem próprio (`sampling_interval_seconds`) **apenas para a comparação**. Por padrão ambos são nulos e a comparação herda a amostragem global de vídeo, preservando a cobertura temporal pericial.

Em execuções desse tipo, a pasta `comparison/` passa a incluir artefatos como:

- `face_set_comparison.json`
- `face_set_comparison_matches.csv`
- `face_set_comparison_summary.txt`
- `face_set_comparison_calibration_scores.csv`
- `face_set_comparison_calibration_model.json`

## Processamento em múltiplos PCs/instâncias

O aplicativo agora pode dividir o lote entre várias instâncias, inspirado no modelo de coordenação do projeto `verificacao_edicao`.

Para isso:

1. todas as estações devem apontar para o mesmo diretório de evidências compartilhado;
2. o modo `Distribuição` deve estar ativado em todas as instâncias;
3. todas devem usar o mesmo `Identificador da execução`;
4. cada instância tentará assumir apenas arquivos livres, usando manifesto global, locks por arquivo e heartbeat de nó;
5. quando o último item do lote for concluído, uma das instâncias consolidará clustering, busca e relatório final.
6. antes da consolidação final, os parciais podem ser validados quanto à integridade e, quando necessário, reprocessados automaticamente;
7. a ação `Monitor` gera um resumo textual e um JSON com nós observados, locks, parciais íntegros e problemas detectados.

Em ambiente Windows, prefira caminho UNC ou outro mapeamento idêntico em todas as máquinas para evitar que cada estação enxergue a execução compartilhada em pastas diferentes.

## Diretório de trabalho

Ao iniciar `Criar inventário`, `Busca por faces` ou `Comparar conjuntos`, a interface solicita um diretório de trabalho separado da pasta de evidências. Esse diretório recebe logs, relatórios, arquivos de monitoramento e artefatos derivados.

Quando a opção de cópia temporária local está ativada, cada imagem ou vídeo é copiado temporariamente para o disco local da estação antes da decodificação. O hash SHA-512 continua sendo calculado a partir do arquivo original.

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
- geração de relatório;
- busca por múltiplas faces;
- comparação entre conjuntos com calibração LR.

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

## Versionamento automático

O repositório publicado no GitHub agora pode atualizar a versão do aplicativo automaticamente.

- a origem da versão continua em `src/inventario_faces/__init__.py`;
- o fallback do instalador em `build/inventario_faces.iss` é mantido sincronizado;
- todo `push` para `main` no GitHub dispara o workflow `.github/workflows/auto-bump-version.yml`;
- esse workflow incrementa automaticamente o `patch` da versão, valida os arquivos de metadados e grava um commit automático de bump.

Para atualizar manualmente a versão antes de publicar, use:

```powershell
python .\tools\bump_version.py --part patch --write
```

ou defina uma versão explícita:

```powershell
python .\tools\bump_version.py --set 0.3.0 --write
```

Se você fizer um bump manual e não quiser que o GitHub faça um segundo bump automático ao chegar no `main`, inclua `[skip version bump]` na mensagem do commit.

## Observação final

O sistema foi desenhado para apoiar triagem e análise pericial, não para substituir juízo técnico humano.
