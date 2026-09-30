# PCA de jogadores de futebol

## Motivação

A Análise de Componentes Principais (PCA) transforma várias métricas correlacionadas em componentes que resumem seus padrões de variação. A primeira componente principal (PC1) é uma combinação linear das métricas que captura a maior parcela da variância observada nos dados. Assim, ela pode ser usada para construir um índice sintético: em vez de avaliar cada métrica isoladamente, projetamos cada jogador nessa combinação e obtemos uma única pontuação.

Este projeto assume que métricas com maior contribuição para a variância capturada pela PC1 carregam mais informação e, portanto, maior importância para o índice. Essa é uma hipótese de modelagem, não uma verdade universal: variância não equivale necessariamente a qualidade esportiva. Por isso, as métricas são normalizadas antes do PCA, e o índice resultante deve ser entendido como um resumo relativo dos jogadores comparados dentro da mesma posição, não como uma medida absoluta de talento.

Este projeto processa dados do dataset `soulcelestia/fbref-data` e calcula um índice baseado na primeira componente principal (PC1) para jogadores de linha. Os goleiros são ignorados.

## O que o programa faz

- Lê os CSVs por posição e expande a coluna `Attribute Vector` em métricas.
- Calcula `npxG` a partir de `npxG + xAG` e `xAG`.
- Consolida nomes repetidos na mesma posição usando a média das métricas.
- Opcionalmente filtra jogadores pelo mínimo de passes tentados por 90 minutos.
- Normaliza as métricas separadamente por posição, aplica PCA e converte o resultado em um índice de 0 a 10.
- Abre um menu para consultar resultados por posição, ver os maiores e menores índices e pesquisar um jogador pelo nome.

O índice é relativo aos jogadores da mesma posição e aos dados processados. Não deve ser comparado entre posições nem interpretado como uma avaliação definitiva da qualidade do jogador. Como os arquivos não fornecem um identificador único, nomes iguais na mesma posição são consolidados.

## Instalação

Requer Python 3.10 ou superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Como executar

Sem argumentos, o programa baixa o dataset usando `kagglehub`:

```powershell
python pca.py
```

Também é possível usar uma pasta local com os CSVs:

```powershell
python pca.py --data-dir "C:\caminho\para\os\csvs"
```

Para filtrar jogadores de linha por passes tentados por 90 minutos, informe um limite. Sem essa opção, ninguém é removido por essa métrica:

```powershell
python pca.py --min-passes-per-90 50
```

As opções podem ser combinadas:

```powershell
python pca.py --data-dir "C:\caminho\para\os\csvs" --min-passes-per-90 50
```
