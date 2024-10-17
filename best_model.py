import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def prepare_data(df):
    """
    Prepara os dados para análise, incluindo normalização.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df[numeric_cols].copy()
    
    # Handle missing values
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    # Normalize the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=df_numeric.columns,
        index=df_numeric.index
    )
    
    return df_scaled


def apply_dbscan(df_scaled, eps=0.5, min_samples=5):
    """
    Aplica o modelo DBSCAN para detectar anomalias.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    predictions = dbscan.fit_predict(df_scaled)
    
    # Retorna apenas as anomalias (indicadas por -1)
    anomalies = df_scaled[predictions == -1]
    return anomalies


def main(df):
    """
    Função principal para preparar os dados, aplicar o DBSCAN e retornar as anomalias.
    """
    print("Preparando dados...")
    df_scaled = prepare_data(df)
    
    print("Aplicando DBSCAN...")
    anomalies = apply_dbscan(df_scaled)
    
    return anomalies

def get_anomalies_from_indices(df_main, df_anomalies):
    """
    Retorna as linhas do DataFrame principal com base nos índices presentes no DataFrame de anomalias.
    
    :param df_main: DataFrame principal contendo todos os dados.
    :param df_anomalies: DataFrame contendo as anomalias com seus índices.
    :return: DataFrame com as linhas correspondentes às anomalias do DataFrame principal.
    """
    # Obter os índices das anomalias
    indices_anomalias = df_anomalies.index
    
    # Filtrar o DataFrame principal usando os índices de anomalias
    df_anomalias_retornadas = df_main.loc[indices_anomalias]
    
    return df_anomalias_retornadas


if __name__ == "__main__":
    try:
        df = pd.read_csv('dados_processados.csv')
        anomalies = main(df)
        
        print(f"Total de anomalias detectadas: {len(anomalies)}")

        df_main = pd.read_csv("dados_coletados_PNCP_ate_pagina_74_normalize.csv")
        anomalias_retornadas = get_anomalies_from_indices(df_main, anomalies)
        
        # Salvando as anomalias em um arquivo CSV
        anomalies.to_csv('anomalias_detectadas.csv', index=True)
        anomalias_retornadas.to_csv('anomalias_detectadas_completas.csv')
        print("Anomalias salvas em 'anomalias_detectadas.csv'")
        
    except Exception as e:
        print(f"Erro ao carregar ou processar os dados: {e}")

#%%
dados_anomalos = pd.read_csv("anomalias_detectadas.csv", index_col='Unnamed: 0')
dados = pd.read_csv("dados_processados.csv")
mascara = ~dados.index.isin(dados_anomalos.index)
dados_sem_anomalias = dados[mascara]


# %%
