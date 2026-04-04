# Inventario Faces

Aplicativo desktop em Python para Windows voltado a inventário forense assistido de faces em imagens e vídeos, com rastreabilidade completa, hashing SHA-512, extração de características faciais, agrupamento por similaridade, relatório técnico em LaTeX/PDF e DOCX, e empacotamento para distribuição.

Repositório oficial: [https://github.com/demusis/inventario_faces](https://github.com/demusis/inventario_faces)

## Finalidade

O projeto foi concebido para apoio investigativo e pericial na triagem de grandes acervos de imagens e vídeos. O sistema:

- recebe um diretório raiz como entrada;
- percorre recursivamente os arquivos;
- identifica imagens e vídeos suportados;
- calcula hash SHA-512 de todos os arquivos;
- detecta faces e extrai vetores faciais;
- agrupa ocorrências por possível indivíduo;
- sugere correlações entre grupos;
- gera inventário estruturado;
- emite relatório técnico em `.tex`, `.pdf` e `.docx`.

## Avisos importantes

- O sistema não realiza identificação conclusiva de indivíduos.
- Os resultados são probabilísticos e exigem revisão humana qualificada.
- O processamento é auditável e reprodutível, mas depende da calibração adequada dos parâmetros.
- Recomenda-se executar em Python 3.11 ou 3.12 para melhor compatibilidade com InsightFace.
- A coleta de características técnicas via MediaInfo depende do executável `mediainfo.exe`, que pode ser localizado pelo `PATH` do sistema ou configurado manualmente na interface.

## Principais recursos

- interface desktop com PySide6;
- configurações persistentes entre execuções;
- ajuda técnica detalhada para cada parâmetro;
- suporte a imagens e vídeos com amostragem temporal configurável;
- seleção de faces por qualidade mínima e tamanho mínimo;
- relatório com galerias comparativas e anexo técnico;
- estatísticas de tamanho facial antes e depois dos filtros;
- extração opcional de características de mídia via MediaInfo;
- exportação de artefatos estruturados em JSON e CSV;
- geração de executável com PyInstaller;
- geração de instalador para Windows com Inno Setup.

## Estrutura do projeto

```text
config/
build/
src/inventario_faces/
  domain/
  services/
  infrastructure/
  gui/
  reporting/
  utils/
tests/
```

## Requisitos

- Windows 10 ou superior
- Python 3.11 ou 3.12
- LaTeX com `pdflatex` disponível no sistema para geração automática do PDF
- MediaInfo opcional para enriquecimento do anexo técnico

## Instalação em desenvolvimento

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Para a análise facial real com InsightFace:

```powershell
pip install "insightface>=0.7.3"
```

Se o `InsightFace` falhar por compilação nativa, instale o conjunto de compilação C++ do Visual Studio Build Tools.

## Execução

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
.\.venv\Scripts\Activate.ps1
python -m inventario_faces
```

Na interface:

1. selecione a pasta de entrada;
2. ajuste parâmetros em `Configurações` se necessário;
3. clique em `Executar`;
4. abra o relatório ao final pelo botão `Abrir Relatório`.

## Saídas geradas

Cada execução cria um diretório derivado dentro do acervo analisado, contendo:

- inventário estruturado;
- recortes faciais;
- quadros anotados;
- registros de execução;
- relatório `.tex`;
- relatório `.pdf`;
- relatório `.docx`.

## Testes

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

## Empacotamento

Para gerar o executável e o instalador:

```powershell
cd "D:\Meu Drive\Em processamento\inventario_faces"
powershell -ExecutionPolicy Bypass -File .\build\build.ps1
```

Artefatos esperados:

- executável empacotado em `dist\InventarioFaces\`
- instalador em `dist\installer\`

## Tecnologias principais

- Python 3.11+
- PySide6
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy
- LaTeX (`pdflatex`)
- python-docx
- PyInstaller
- Inno Setup

## Licenciamento e publicação

O repositório está preparado para distribuição aberta do código-fonte. Antes de publicar versões operacionais, recomenda-se revisar:

- política institucional de uso investigativo;
- termos de distribuição;
- dependências opcionais de terceiros;
- estratégia de versionamento e changelog.
