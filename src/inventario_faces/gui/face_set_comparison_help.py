from __future__ import annotations

from html import escape

from inventario_faces.domain.config import AppConfig


def _help_det_size(config: AppConfig) -> str:
    return (
        f"{config.face_model.det_size[0]}x{config.face_model.det_size[1]}"
        if config.face_model.det_size is not None
        else "resolução original do quadro"
    )


def _help_providers(config: AppConfig) -> str:
    return (
        ", ".join(config.face_model.providers)
        if config.face_model.providers
        else "seleção automática com preferência por GPU e fallback para CPU"
    )


def build_face_set_comparison_help_html(config: AppConfig) -> str:
    image_extensions = ", ".join(config.media.image_extensions)
    det_size = _help_det_size(config)
    providers = _help_providers(config)
    return f"""
<html>
<head>
<style>
body {{
    font-family: 'Segoe UI', sans-serif;
    color: #0f172a;
    line-height: 1.45;
}}
h1 {{
    font-size: 22px;
    color: #0f172a;
    margin: 0 0 10px 0;
}}
h2 {{
    font-size: 17px;
    color: #0f766e;
    margin: 18px 0 6px 0;
}}
h3 {{
    font-size: 14px;
    color: #334155;
    margin: 14px 0 4px 0;
}}
p, li {{
    font-size: 13px;
}}
code {{
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 4px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 8px;
}}
th, td {{
    border: 1px solid #d7e0ea;
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
}}
th {{
    background: #eef4fa;
}}
</style>
</head>
<body>
<h1>Ajuda da comparação entre grupos faciais</h1>
<p>
Esta janela compara dois conjuntos de imagens faciais, normalmente um conjunto de referência
(<b>Padrão</b>) e um conjunto sob exame (<b>Questionado</b>). O sistema processa as imagens,
seleciona faces elegíveis, compara as representações faciais e organiza o resultado em saídas auditáveis.
</p>

<h2>Objetivo da janela</h2>
<ul>
    <li>Comparar diretamente dois grupos de imagens faciais.</li>
    <li>Classificar os pares por nível de interesse: atribuição, candidata ou abaixo do limiar.</li>
    <li>Exibir medidas descritivas e inferenciais quando houver repetição e variabilidade suficientes.</li>
    <li>Aplicar razão de verossimilhança calibrada quando houver base LR ou modelo LR salvo.</li>
    <li>Gerar artefatos exportáveis, logs e trilha de auditoria da execução.</li>
</ul>

<h2>Fluxo recomendado de uso</h2>
<ol>
    <li>Adicione as imagens do grupo <b>Padrão</b>.</li>
    <li>Adicione as imagens do grupo <b>Questionado</b>.</li>
    <li>Confirme o <b>Diretório de trabalho</b>.</li>
    <li>Se necessário, informe a <b>Base de calibração LR</b> ou carregue um <b>Modelo de calibração LR</b>.</li>
    <li>Clique em <b>Comparar conjuntos</b>.</li>
    <li>Ao final, revise <b>Resumo estatístico</b>, <b>Correspondências</b> e, se houver, <b>Razão de verossimilhança</b>.</li>
    <li>Se a calibração for útil para execuções futuras, preserve o JSON salvo automaticamente ou use <b>Salvar modelo LR</b>.</li>
</ol>

<h2>Controles principais</h2>
<table>
    <tr><th>Controle</th><th>Função</th><th>Orientação</th></tr>
    <tr>
        <td><b>Padrão</b></td>
        <td>Grupo de referência usado como fonte das faces do conjunto A.</td>
        <td>Use imagens representativas e bem documentadas. Extensões aceitas nesta configuração: <code>{escape(image_extensions)}</code>.</td>
    </tr>
    <tr>
        <td><b>Questionado</b></td>
        <td>Grupo examinado, tratado como conjunto B na comparação.</td>
        <td>Mantenha apenas material pertinente à hipótese examinada para evitar inflar o número de pares sem necessidade.</td>
    </tr>
    <tr>
        <td><b>Adicionar imagens</b></td>
        <td>Inclui novos arquivos no grupo.</td>
        <td>Ideal para montar o conjunto de forma incremental.</td>
    </tr>
    <tr>
        <td><b>Remover selecionadas</b></td>
        <td>Retira apenas os itens marcados na lista.</td>
        <td>Útil para limpar erros de seleção sem reiniciar tudo.</td>
    </tr>
    <tr>
        <td><b>Limpar</b></td>
        <td>Esvazia completamente o grupo correspondente.</td>
        <td>Use quando quiser reiniciar a montagem do conjunto.</td>
    </tr>
    <tr>
        <td><b>Diretório de trabalho</b></td>
        <td>Local em que a execução grava logs, tabelas, gráficos, JSONs, ZIP e demais artefatos.</td>
        <td>Escolha uma pasta com espaço suficiente e preservável para auditoria posterior.</td>
    </tr>
    <tr>
        <td><b>Base de calibração LR (opcional)</b></td>
        <td>Diretório com uma subpasta por identidade rotulada, usado para estimar as distribuições de mesma origem e de origem distinta.</td>
        <td>É a opção mais completa, mas também a mais custosa em tempo de processamento.</td>
    </tr>
    <tr>
        <td><b>Modelo de calibração LR (opcional)</b></td>
        <td>Arquivo JSON com o modelo LR já calculado em execução anterior.</td>
        <td>Se a base de calibração não mudou, esta é a forma recomendada de reaproveitar a LR sem recalcular tudo.</td>
    </tr>
    <tr>
        <td><b>Comparar conjuntos</b></td>
        <td>Inicia o pipeline da comparação.</td>
        <td>Durante a execução, os controles de entrada são bloqueados para manter a consistência do procedimento.</td>
    </tr>
    <tr>
        <td><b>Ajuda</b></td>
        <td>Abre este painel explicativo.</td>
        <td>Use-o como referência rápida operacional e interpretativa.</td>
    </tr>
    <tr>
        <td><b>Exportar ZIP</b></td>
        <td>Compacta o diretório de execução concluída.</td>
        <td>Útil para preservação do procedimento e circulação controlada dos artefatos.</td>
    </tr>
    <tr>
        <td><b>Abrir execução</b></td>
        <td>Abre a pasta da execução atual.</td>
        <td>Permite acessar diretamente os arquivos exportados, logs e modelos LR.</td>
    </tr>
</table>

<h2>Barra de resultados</h2>
<table>
    <tr><th>Controle</th><th>Uso</th><th>Leitura</th></tr>
    <tr>
        <td><b>Reamostragens</b></td>
        <td>Quantidade de amostras bootstrap usada nos intervalos de confiança.</td>
        <td>Valores maiores tendem a estabilizar a inferência, porém aumentam o custo computacional.</td>
    </tr>
    <tr>
        <td><b>Significância (%)</b></td>
        <td>Nível de significância aplicado à inferência bootstrap.</td>
        <td>Por exemplo, 5% produz um intervalo bilateral aproximado de 95%.</td>
    </tr>
    <tr>
        <td><b>Resumo estatístico</b></td>
        <td>Mostra contagens, média, mediana, quartis, IC bootstrap, teste U de Mann-Whitney e procedimento.</td>
        <td>Resume o conjunto inteiro de pares comparados e também compara, de forma não paramétrica, a qualidade facial entre Padrão e Questionado.</td>
    </tr>
    <tr>
        <td><b>Distribuição</b></td>
        <td>Exibe curvas de densidade não paramétricas por classe decisória, marcadores dos limiares e uma linha tracejada no melhor escore observado nesta janela.</td>
        <td>Ajuda a visualizar dispersão, separação entre classes e a posição do melhor par dentro da distribuição geral, sem aplicar a calibração LR.</td>
    </tr>
    <tr>
        <td><b>Entradas processadas</b></td>
        <td>Lista arquivo a arquivo com detectadas, selecionadas, tracks, keyframes e estado.</td>
        <td>É a saída primária para auditoria operacional e checagem de aproveitamento das imagens.</td>
    </tr>
    <tr>
        <td><b>Correspondências</b></td>
        <td>Mostra o ranking dos pares comparados.</td>
        <td>É a saída central para revisão pericial; deve ser lida junto com a inspeção visual.</td>
    </tr>
    <tr>
        <td><b>Malha biométrica</b></td>
        <td>Mostra a tabela de correspondências com as imagens/derivados associados ao par selecionado.</td>
        <td>Serve para revisar contexto, recorte, qualidade, keyframe e coerência visual do par.</td>
    </tr>
    <tr>
        <td><b>Razão de verossimilhança</b></td>
        <td>Exibe estado da calibração LR, histogramas brutos, densidades H1/H2 e a tabela com <code>LR</code>, <code>log10(LR)</code> e evidência.</td>
        <td>Use quando a execução estiver calibrada e a base ou modelo LR forem tecnicamente compatíveis com o caso. Nesta janela, a linha azul acompanha a linha selecionada na tabela.</td>
    </tr>
    <tr>
        <td><b>Salvar modelo LR</b></td>
        <td>Salva manualmente o modelo LR corrente em JSON.</td>
        <td>Disponível apenas depois de uma execução com calibração LR.</td>
    </tr>
</table>

<h2>Pipeline técnico resumido</h2>
<h3>1. Preparação e leitura das imagens</h3>
<p>
As imagens dos dois grupos são abertas individualmente e registradas como entradas processadas. A comparação não trabalha
com a “pasta inteira” como uma única unidade, mas com as faces elegíveis encontradas em cada arquivo.
</p>

<h3>2. Detecção, filtros e extração facial</h3>
<p>
O backend atual é <code>{escape(config.face_model.backend)}</code>, com modelo
<code>{escape(config.face_model.model_name)}</code>, tamanho de detecção
<code>{escape(det_size)}</code>, qualidade mínima
<code>{config.face_model.minimum_face_quality:.2f}</code>, tamanho mínimo de face
<code>{config.face_model.minimum_face_size_pixels}px</code>, <code>ctx_id={config.face_model.ctx_id}</code>
e providers configurados como <code>{escape(providers)}</code>.
</p>
<p>
Cada imagem passa por detecção facial, filtros de qualidade e tamanho, seleção de ocorrências elegíveis e geração de embeddings.
Faces inelegíveis podem ser descartadas antes da comparação propriamente dita.
</p>

<h3>3. Comparação entre Padrão e Questionado</h3>
<p>
Depois da seleção das faces válidas, o sistema compara cada face elegível do grupo Padrão com cada face elegível do grupo
Questionado. Em termos práticos, o número de comparações cresce aproximadamente com
<code>faces_padrão × faces_questionado</code>.
</p>

<h3>4. Similaridade e classes decisórias</h3>
<p>
Cada par recebe uma similaridade facial. Em seguida, o sistema o classifica segundo os limiares de decisão da configuração:
</p>
<ul>
    <li><b>Atribuição</b>: resultado que atinge o limiar principal de atribuição.</li>
    <li><b>Candidata</b>: resultado abaixo da atribuição, mas acima do limiar de sugestão investigativa.</li>
    <li><b>Abaixo do limiar</b>: resultado que não atingiu o patamar mínimo de interesse definido.</li>
</ul>
<p>
Na configuração carregada nesta sessão, o limiar de atribuição é
<code>{config.clustering.assignment_similarity:.2f}</code> e o limiar de sugestão é
<code>{config.clustering.candidate_similarity:.2f}</code>.
</p>

<h3>5. Como as estatísticas finais são obtidas quando há várias imagens</h3>
<p>
O sistema não escolhe previamente uma “imagem final” do grupo <b>Padrão</b> e outra do grupo <b>Questionado</b>. Primeiro ele
reúne todas as faces selecionadas do Padrão e todas as faces selecionadas do Questionado. Depois compara cada face elegível do
Padrão com cada face elegível do Questionado.
</p>
<p>
Assim, se o grupo Padrão gerar 3 faces selecionadas e o grupo Questionado gerar 4, o resultado final terá
<code>3 × 4 = 12</code> scores. O resumo estatístico é calculado sobre essa coleção inteira de scores:
</p>
<ul>
    <li><b>Comparações</b>: total de pares efetivamente comparados.</li>
    <li><b>Atribuições</b> e <b>Candidatas</b>: quantos pares ultrapassaram cada limiar.</li>
    <li><b>Melhor similaridade</b>: maior score do ranking.</li>
    <li><b>Média, mediana, quartis, desvio e IC</b>: calculados sobre todos os pares, e não apenas sobre o primeiro colocado.</li>
</ul>
<p>
Por isso, quando há muitas imagens ou muitas faces em ambos os lados, o <b>Resumo estatístico</b> descreve o comportamento
global do conjunto de pares, enquanto a janela de <b>Correspondências</b> mostra cada par individualmente, ordenado do maior
para o menor score.
</p>
<p>
Se houver apenas um único par elegível, média, mediana, quartis e melhor score coincidem no mesmo valor. Quando a LR estiver
disponível, a média e a mediana de <code>log10(LR)</code> seguem essa mesma lógica, mas usando todos os pares calibrados.
</p>

<h3>6. Inferência estatística</h3>
<p>
Se houver número suficiente de repetições e variabilidade entre os valores, a janela calcula média, mediana, quartis,
intervalo de confiança bootstrap e curvas de densidade não paramétricas. Se a amostra for pequena demais ou quase constante, a interface informa
que a inferência não pôde ser apresentada.
</p>
<p>
O intervalo de confiança da média usa <b>bootstrap percentílico não paramétrico</b>: a rotina reamostra, com reposição,
os scores observados e extrai os quantis de acordo com a significância configurada na barra de resultados.
</p>
<p>
O mesmo resumo também executa um <b>teste U de Mann-Whitney bilateral</b> para comparar a distribuição de
<b>qualidade facial</b> das faces selecionadas no grupo Padrão com a do grupo Questionado. Esse teste é
<b>não paramétrico</b>, trabalha com postos em vez de assumir normalidade e ajuda a verificar se um grupo
chegou à comparação com material sistematicamente melhor ou pior do que o outro.
</p>

<h3>7. Calibração por razão de verossimilhança (LR)</h3>
<p>
Quando uma base rotulada ou um modelo LR salvo é informado, o sistema pode calibrar a interpretação do score. Em vez de
usar apenas a similaridade bruta, ele estima o quanto aquele valor é compatível com:
</p>
<ul>
    <li><b>H1</b>: mesma origem.</li>
    <li><b>H2</b>: origem distinta.</li>
</ul>
<p>
Se a base rotulada for usada, o sistema gera scores de mesma origem e origem distinta a partir das subpastas-identidade,
ajusta densidades e calcula <code>LR</code> e <code>log10(LR)</code> para os pares do caso.
Se um modelo salvo for carregado, essa etapa é reaproveitada sem recalcular toda a base.
</p>
<p><b>Importante:</b> se a base rotulada e o modelo salvo forem informados ao mesmo tempo, o <b>modelo salvo tem prioridade</b>.</p>

<h2>Interpretação dos campos principais</h2>
<h3>Similaridade</h3>
<p>
É a medida comparativa bruta entre os embeddings das duas faces. Similaridade alta indica maior proximidade no espaço do
modelo, mas não equivale, isoladamente, a identificação conclusiva.
</p>

<h3>Classe</h3>
<p>
A classe mostra a posição do par em relação aos limiares. <b>Atribuição</b> representa um resultado mais forte do que
<b>Candidata</b>, porém ambos exigem revisão humana. <b>Abaixo do limiar</b> indica que o par não atingiu o patamar mínimo
de interesse configurado.
</p>

<h3>Qualidade</h3>
<p>
As colunas de qualidade ajudam a julgar o valor prático do par. Um score alto vindo de faces ruins pede cautela extra; um
score moderado vindo de material ruim também pode estar artificialmente deprimido pela qualidade da entrada.
</p>

<h3>Detecção</h3>
<p>
O campo <b>Detecção</b> representa a confiança do detector facial de que o retângulo extraído realmente contém uma face.
Ele está ligado à etapa de localização da face, e não à comparação biométrica entre Padrão e Questionado.
</p>
<p>
Por isso, <b>Detecção</b> não deve ser lida como probabilidade de identidade e nem como força da evidência. Um valor alto de
detecção indica apenas que a face foi localizada com boa confiança. Ainda assim, a face pode ter qualidade limitada ou gerar
baixa similaridade com o par comparado.
</p>
<p>
Em termos práticos: <b>Detecção</b> responde “o detector encontrou uma face confiável aqui?”, <b>Qualidade</b> responde
“essa face está boa para análise?” e <b>Similaridade</b> responde “quão próximos ficaram os embeddings dessas duas faces?”.
</p>

<h3>Resumo estatístico</h3>
<p>
Use esta janela para verificar quantos pares foram comparados, quantos ultrapassaram os limiares, se houve suporte
suficiente para inferência e como a distribuição geral se comporta. Quando houver várias faces em cada lado, esses números
representam o conjunto inteiro de pares comparados, e não apenas o melhor item do ranking.
</p>
<p>
Além disso, o resumo informa o resultado do <b>U de Mann-Whitney</b> sobre a qualidade facial das faces selecionadas.
Se o <code>p-valor</code> ficar abaixo da significância configurada, há evidência estatística de que os dois grupos
não chegaram à comparação com a mesma distribuição de qualidade. Isso é útil para contextualizar assimetrias operacionais
entre Padrão e Questionado antes de interpretar os scores e a LR.
</p>

<h3>Distribuição</h3>
<p>
As curvas de densidade ajudam a enxergar concentração, separação e dispersão dos scores. Curvas muito sobrepostas sugerem menor
separação prática; curvas mais apartadas indicam comportamento mais estável dos resultados.
</p>
<p>
No popup de <b>Distribuição</b>, a linha tracejada azul marca o <b>melhor escore observado</b> na execução atual. Ela serve para
localizar o primeiro item do ranking dentro da distribuição global de scores.
</p>

<h3>Entradas processadas</h3>
<p>
É a visão mais importante para auditoria do processamento. Antes de interpretar qualquer ranking, vale conferir se as
imagens relevantes foram realmente aproveitadas, quantas faces foram detectadas e se houve erro em algum arquivo.
</p>

<h3>Correspondências</h3>
<p>
O ranking organiza os pares mais relevantes da comparação. Em geral, a revisão começa pelos primeiros itens, mas a ordem
não substitui a análise pericial do contexto e da qualidade do material.
</p>

<h3>Malha biométrica</h3>
<p>
Esta visualização integra números e imagem. Ela serve para inspeção qualitativa do par selecionado, verificando contexto,
recorte, qualidade, keyframe e coerência geral do material exibido.
</p>

<h3>LR, log10(LR) e evidência</h3>
<ul>
    <li><code>LR &gt; 1</code> favorece a hipótese de mesma origem.</li>
    <li><code>LR &lt; 1</code> favorece a hipótese de origem distinta.</li>
    <li><code>log10(LR) = 0</code> é aproximadamente neutro entre H1 e H2.</li>
    <li><code>log10(LR) &gt; 0</code> favorece mesma origem; quanto maior, mais forte o suporte.</li>
    <li><code>log10(LR) &lt; 0</code> favorece origem distinta; quanto menor, mais forte o suporte para H2.</li>
</ul>
<p>
O rótulo textual de <b>Evidência</b> resume essa magnitude. Ainda assim, a força da LR depende da adequação da base ou do
modelo LR ao pipeline realmente usado no caso.
</p>
<p>
No popup de <b>Razão de verossimilhança</b>, a linha tracejada azul marca o <b>score do confronto selecionado na tabela</b>. O valor de
<code>LR</code> é obtido pela razão entre a <b>altura da curva verde (H1)</b> e a <b>altura da curva vermelha (H2)</b>
exatamente nesse ponto do eixo horizontal. Em termos práticos: <code>LR = f(score|H1) / f(score|H2)</code>.
</p>

<h2>Quando reutilizar um modelo LR salvo</h2>
<ul>
    <li>Quando a base de calibração é a mesma.</li>
    <li>Quando backend, modelo facial, thresholds e condições operacionais continuam compatíveis.</li>
    <li>Quando o objetivo é comparar novos grupos Padrão/Questionado sem recalcular a calibração inteira.</li>
</ul>
<p>
Se houver mudança material na base rotulada, no pipeline ou na configuração, o mais prudente é recalibrar.
</p>

<h2>Cautelas periciais e boas práticas</h2>
<ul>
    <li>Não trate o resultado automático como prova conclusiva de identidade.</li>
    <li>Combine leitura numérica, revisão visual e contexto do caso.</li>
    <li>Leve em conta a qualidade das faces antes de valorar a força de um par.</li>
    <li>Ao usar LR, confirme se a calibração estava realmente disponível e com suporte suficiente.</li>
    <li>Preserve o diretório de execução ou o ZIP exportado como parte da trilha de auditoria.</li>
</ul>

<h2>Arquivos gerados e rastreabilidade</h2>
<p>
Cada execução grava um diretório próprio com logs, tabelas, resumos, artefatos de imagem e, quando aplicável, arquivos da
calibração LR. O botão <b>Abrir execução</b> leva diretamente a essa pasta, e <b>Exportar ZIP</b> permite empacotá-la.
</p>

<h2>Se a distribuição ou a LR não aparecerem</h2>
<p>
As saídas inferenciais podem ficar indisponíveis quando houver amostra insuficiente, variabilidade muito baixa, calibração
ausente ou base/modelo LR sem suporte suficiente. Nesses casos, os popups e o log informam o motivo de forma explícita.
</p>
</body>
</html>
"""
