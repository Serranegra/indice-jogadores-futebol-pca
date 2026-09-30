import argparse
import ast
from pathlib import Path
from typing import Any

import kagglehub
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


DATASET_ID = "soulcelestia/fbref-data"
COLUNA_VETOR = "Attribute Vector"
ARQUIVO_GOLEIROS = "GoalKeepers.csv"
ARQUIVO_IGNORADO = "NAME_DB.csv"
NOMES_POSICOES = {
    "AtMid_Wingers.csv": "Meias ofensivos e pontas",
    "CenterBacks.csv": "Zagueiros",
    "Forwards.csv": "Atacantes",
    "FullBacks.csv": "Laterais",
    "Midfielders.csv": "Meio-campistas",
}

COLUNAS_NAO_GK_VETOR = (
    "Non-Penalty Goals",
    "Non-Penalty xG",
    "Shots Total",
    "Assists",
    "xAG",
    "npxG + xAG",
    "Shot-Creating Actions",
    "Passes Attempted",
    "Pass Completion %",
    "Progressive Passes",
    "Progressive Carries",
    "Successful Take-Ons",
    "Touches (Att Pen)",
    "Progressive Passes Rec",
    "Tackles",
    "Interceptions",
    "Blocks",
    "Clearances",
    "Aerials won",
)

COLUNAS_NAO_GK = tuple(
    coluna for coluna in COLUNAS_NAO_GK_VETOR if coluna != "npxG + xAG"
) + ("npxG",)

def converter_seguro(valor: Any) -> Any:
    """Converte uma string de lista para Python sem interromper a leitura."""
    
    if isinstance(valor, str):
        try:
            return ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            return []
    return valor


def obter_colunas() -> tuple[str, ...]:
    """Retorna as métricas usadas no PCA dos jogadores de linha."""

    return COLUNAS_NAO_GK


def obter_nome_posicao(df: pd.DataFrame) -> str:
    """Obtém o nome legível da posição a partir do CSV de origem."""

    arquivo = df["arquivo_origem"].iloc[0]
    return NOMES_POSICOES.get(arquivo, Path(arquivo).stem.replace("_", " "))


def expandir_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Expande o vetor de métricas e calcula npxG a partir dos seus componentes."""

    df = df.copy()
    vetores = df[COLUNA_VETOR].apply(converter_seguro)
    atributos = pd.DataFrame(
        vetores.tolist(), index=df.index, columns=COLUNAS_NAO_GK_VETOR
    )

    atributos["npxG"] = atributos["npxG + xAG"] - atributos["xAG"]
    atributos = atributos.drop(columns=["npxG + xAG"])
    atributos = atributos.loc[:, COLUNAS_NAO_GK]

    return pd.concat([df.drop(columns=[COLUNA_VETOR]), atributos], axis=1)


def carregar_dados(pasta_destino: str | Path) -> list[pd.DataFrame]:
    """Lê os CSVs de jogadores de linha e ignora goleiros e arquivos auxiliares."""

    dataframes = []

    for arquivo in Path(pasta_destino).rglob("*.csv"):
        if arquivo.name == ARQUIVO_GOLEIROS:
            print("Ignorando GoalKeepers.csv: goleiros fora do PCA.")
            continue
        if arquivo.name == ARQUIVO_IGNORADO:
            continue

        try:
            df = pd.read_csv(arquivo)
        except (
            OSError,
            UnicodeDecodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as erro:
            print(f"Erro ao ler {arquivo.name}: {erro}")
            continue

        if df.empty:
            print(f"Aviso: {arquivo.name} está vazio; ignorando.")
            continue

        df["arquivo_origem"] = arquivo.name
        if COLUNA_VETOR not in df.columns:
            print(f"Aviso: {arquivo.name} não possui '{COLUNA_VETOR}'; ignorando.")
            continue

        dataframes.append(expandir_atributos(df))
        print(f"Lido: {arquivo.name} | Linhas: {len(df)}")

    if not dataframes:
        raise ValueError("Nenhum CSV com vetores de atributos foi encontrado.")

    return dataframes


def filtrar_jogadores(
    df: pd.DataFrame, min_passes_per_90: float | None = None
) -> pd.DataFrame:
    """Filtra passes/90 e agrupa por nome, assumido como identidade na posição."""

    if "Name" not in df.columns:
        raise ValueError("O dataframe não possui a coluna 'Name'.")
    if min_passes_per_90 is not None and min_passes_per_90 < 0:
        raise ValueError("O limite mínimo de passes/90 não pode ser negativo.")

    df_filtrado = df.copy()
    nomes = df_filtrado["Name"].astype("string").str.strip()
    nomes_validos = nomes.notna() & nomes.ne("")
    removidos_sem_nome = int((~nomes_validos).sum())
    df_filtrado = df_filtrado.loc[nomes_validos].copy()
    df_filtrado["Name"] = nomes.loc[nomes_validos]

    arquivo = df["arquivo_origem"].iloc[0]
    if min_passes_per_90 is not None:
        if "Passes Attempted" not in df_filtrado.columns:
            raise ValueError(f"{arquivo} não possui a métrica 'Passes Attempted'.")
        passes = pd.to_numeric(df_filtrado["Passes Attempted"], errors="coerce")
        quantidade_anterior = len(df_filtrado)
        df_filtrado = df_filtrado.loc[passes >= min_passes_per_90].copy()
        removidos_por_filtro = quantidade_anterior - len(df_filtrado)
        if removidos_por_filtro:
            print(
                f"{obter_nome_posicao(df)}: {removidos_por_filtro} jogadores "
                f"abaixo de {min_passes_per_90:g} passes/90 removidos."
            )
    if removidos_sem_nome:
        print(f"{removidos_sem_nome} linhas sem nome foram removidas.")
    if df_filtrado.empty:
        return df_filtrado

    df_filtrado["_chave_jogador"] = df_filtrado["Name"].str.casefold()
    duplicatas = int(df_filtrado["_chave_jogador"].duplicated().sum())
    colunas_metricas = list(obter_colunas())

    metadados = (
        df_filtrado.drop(columns=colunas_metricas)
        .drop_duplicates(subset="_chave_jogador")
        .set_index("_chave_jogador")
    )
    medias = df_filtrado.groupby("_chave_jogador", sort=False)[
        colunas_metricas
    ].mean()
    df_limpo = metadados.join(medias).reset_index(drop=True)

    if duplicatas:
        print(
            f"{obter_nome_posicao(df)}: {duplicatas} linhas duplicadas "
            "consolidadas pela média das métricas."
        )
    return df_limpo

def normalizar_dados(dataframes: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Escala as métricas por posição; notas finais não são comparáveis entre posições."""

    normalizados = []

    for df in dataframes:
        colunas = obter_colunas()
        df_normalizado = df.copy()
        scaler = MinMaxScaler()
        df_normalizado.loc[:, colunas] = scaler.fit_transform(df.loc[:, colunas])
        normalizados.append(df_normalizado)

    return normalizados


def adicionar_indice_pca(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula PC1 e uma nota relativa de 0 a 10 para uma posição."""

    df = df.copy()
    colunas = obter_colunas()
    dados = df.loc[:, colunas].fillna(0)

    pca = PCA(n_components=1)
    componente = pca.fit_transform(dados)
    pesos_pc1 = pca.components_[0].copy()

    # O sinal do PC é arbitrário; esta convenção prioriza pesos com soma positiva.
    if pesos_pc1.sum() < 0:
        componente = -componente
        pesos_pc1 = -pesos_pc1

    scaler_nota = MinMaxScaler(feature_range=(0, 10))
    df["indice_pca"] = scaler_nota.fit_transform(componente).round(2)
    df.attrs["pesos_pc1"] = tuple(zip(colunas, pesos_pc1.tolist()))
    df.attrs["posicao"] = obter_nome_posicao(df)
    return df


def executar_pipeline(
    pasta_destino: str | Path | None = None,
    min_passes_per_90: float | None = None,
) -> list[pd.DataFrame]:
    """Executa carregamento, limpeza, normalização e PCA para cada posição."""

    if pasta_destino is None:
        pasta_destino = kagglehub.dataset_download(DATASET_ID)

    dataframes = carregar_dados(pasta_destino)
    filtrados = [
        filtrar_jogadores(df, min_passes_per_90) for df in dataframes
    ]
    filtrados = [df for df in filtrados if not df.empty]
    if not filtrados:
        raise ValueError("Nenhum jogador restou após a filtragem.")

    normalizados = normalizar_dados(filtrados)
    return [adicionar_indice_pca(df) for df in normalizados]


def exibir_resultados_posicao(df: pd.DataFrame) -> None:
    """Exibe pesos do PC1 e o ranking completo da posição."""

    posicao = obter_nome_posicao(df)
    print(f"\n=== Resultados PCA: {posicao} ===")

    dados_pesos = df.attrs.get("pesos_pc1")
    if dados_pesos is not None:
        pesos = pd.DataFrame(dados_pesos, columns=["metrica", "peso_pc1"])
        print("\nPesos das métricas no PC1:")
        print(pesos.sort_values("peso_pc1", ascending=False).to_string(index=False))

    ranking = df.loc[:, ["Name", "indice_pca"]].sort_values(
        "indice_pca", ascending=False
    )
    print("\nRanking completo:")
    print(ranking.to_string(index=False))


def exibir_top_e_piores(df: pd.DataFrame, limite: int = 10) -> None:
    """Exibe os maiores e menores índices PCA da posição."""

    posicao = obter_nome_posicao(df)
    ranking = df.loc[:, ["Name", "indice_pca"]]

    print(f"\n=== {posicao}: top {limite} ===")
    print(ranking.nlargest(limite, "indice_pca").to_string(index=False))
    print(f"\n=== {posicao}: {limite} menores índices ===")
    print(ranking.nsmallest(limite, "indice_pca").to_string(index=False))


def buscar_jogador(
    resultados: list[pd.DataFrame], consulta: str
) -> pd.DataFrame:
    """Busca um jogador e retorna índice e colocação dentro de cada posição.

    Tenta primeiro o nome completo, sem diferenciar maiúsculas e minúsculas.
    Se não houver correspondência exata, procura nomes que contenham a consulta.
    Empates no índice compartilham a mesma colocação.
    """
    consulta_normalizada = consulta.strip().casefold()
    colunas_saida = [
        "Jogador",
        "Posição",
        "Índice PCA",
        "Colocação",
        "Total na posição",
        "Correspondência",
    ]
    if not consulta_normalizada:
        return pd.DataFrame(columns=colunas_saida)

    exatos = []
    parciais = []
    for df in resultados:
        nomes = df["Name"].astype("string").str.strip().str.casefold()
        mascara_exata = nomes.eq(consulta_normalizada).fillna(False)
        mascara_parcial = nomes.str.contains(
            consulta_normalizada, regex=False, na=False
        )
        colocacoes = df["indice_pca"].rank(
            method="min", ascending=False
        ).astype(int)
        correspondencias = exatos if mascara_exata.any() else parciais

        for indice in df.index[mascara_exata if mascara_exata.any() else mascara_parcial]:
            correspondencias.append(
                {
                    "Jogador": df.at[indice, "Name"],
                    "Posição": obter_nome_posicao(df),
                    "Índice PCA": df.at[indice, "indice_pca"],
                    "Colocação": int(colocacoes.at[indice]),
                    "Total na posição": len(df),
                    "Correspondência": (
                        "exata" if mascara_exata.any() else "parcial"
                    ),
                }
            )

    correspondencias = exatos if exatos else parciais
    return pd.DataFrame(correspondencias, columns=colunas_saida)


def pesquisar_jogador(resultados: list[pd.DataFrame]) -> None:
    """Solicita um nome e exibe o índice PCA e a colocação encontrada."""
    consulta = input("Nome do jogador: ").strip()
    correspondencias = buscar_jogador(resultados, consulta)

    if correspondencias.empty:
        print(f"Nenhum jogador encontrado para '{consulta}'.")
        return
    if correspondencias["Correspondência"].eq("parcial").all():
        print("Nenhum nome exato encontrado; mostrando correspondências parciais.")

    print(correspondencias.to_string(index=False))


def escolher_posicao(resultados: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Solicita ao usuário uma posição e retorna o respectivo DataFrame."""

    print("\nPosições disponíveis:")
    for indice, df in enumerate(resultados, start=1):
        print(f"{indice}. {obter_nome_posicao(df)}")

    while True:
        escolha = input("Número da posição (ou 0 para voltar): ").strip()
        if escolha == "0":
            return None
        if escolha.isdigit() and 1 <= int(escolha) <= len(resultados):
            return resultados[int(escolha) - 1]
        print("Opção inválida; escolha um dos números listados.")


def exibir_menu(resultados: list[pd.DataFrame]) -> None:
    """Apresenta as opções de consulta dos resultados."""

    while True:
        print("\n=== Consulta dos resultados PCA ===")
        print("1. Resultados completos de uma posição")
        print("2. Top 10 melhores e piores de cada posição")
        print("3. Pesquisar jogador por nome")
        print("0. Sair")
        escolha = input("Escolha uma opção: ").strip()

        if escolha == "0":
            break
        if escolha == "1":
            df = escolher_posicao(resultados)
            if df is not None:
                exibir_resultados_posicao(df)
        elif escolha == "2":
            for df in resultados:
                exibir_top_e_piores(df)
        elif escolha == "3":
            pesquisar_jogador(resultados)
        else:
            print("Opção inválida.")


def main() -> None:
    """Configura argumentos de linha de comando e inicia o menu."""

    parser = argparse.ArgumentParser(
        description="Consulte os índices PCA dos jogadores por posição."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Usa os CSVs de uma pasta local em vez de baixar o dataset.",
    )
    parser.add_argument(
        "--min-passes-per-90",
        type=float,
        default=None,
        help=(
            "Filtra jogadores de linha pelo mínimo de passes tentados por 90; "
            "por padrão, não aplica esse filtro."
        ),
    )
    argumentos = parser.parse_args()

    resultados = executar_pipeline(
        argumentos.data_dir,
        min_passes_per_90=argumentos.min_passes_per_90,
    )
    exibir_menu(resultados)


if __name__ == "__main__":
    main()
