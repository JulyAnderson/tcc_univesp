import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from modelos import prepare_data, apply_isolation_forest, apply_dbscan, apply_lof, apply_kmeans, compare_results

st.set_page_config(layout="wide")

def plotar_resultados(df_escalado, dict_predicoes):
    """Visualiza os resultados."""
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_escalado)

    num_metodos = len(dict_predicoes)
    fig, axes = plt.subplots(1, num_metodos, figsize=(5 * num_metodos, 5))

    for idx, (nome_metodo, predicoes) in enumerate(dict_predicoes.items()):
        scatter = axes[idx].scatter(
            df_pca[:, 0],
            df_pca[:, 1],
            c=predicoes,
            cmap='RdYlBu',
            alpha=0.6
        )
        axes[idx].set_title(f'{nome_metodo}\n{np.sum(predicoes == -1)} anomalias detectadas')
        axes[idx].set_xlabel('PC1')
        axes[idx].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[idx])

    st.pyplot(fig)

def main():
    st.title("Painel de Resultados de Detecção de Anomalias")

    # Carregar dados
    df = pd.read_csv('dados_processados.csv')
    st.write("Prévia dos Dados:")
    st.dataframe(df.head())

    # Preparar dados
    st.write("Preparando dados...")
    df_escalado, df_numerico = prepare_data(df)

    df_scaled_robust, _ = prepare_data(df, robust=True)

    # Aplicar modelos
    dict_resultados = {
        'Isolation Forest': apply_isolation_forest(df_escalado),
        'DBSCAN': apply_dbscan(df_escalado),
        'LOF': apply_lof(df_escalado),
        'K-means': apply_kmeans(df_escalado),
        'DBSCAN Tuning': apply_dbscan(df_scaled_robust, eps=0.5, min_samples=5),
        'K-means Tuning': apply_kmeans(df_scaled_robust, n_clusters=2)
    }

    # Comparação
    st.write("Comparando resultados...")
    comparacao = compare_results(df_numerico, dict_resultados, df_escalado)
    st.write(comparacao)

    # Plotar resultados
    st.write("Visualizando resultados...")
    plotar_resultados(df_escalado, dict_resultados)

    # Visualizar métricas adicionais
    st.write("Visualizando métricas:")
    fig_metrica, ax = plt.subplots()
    sns.barplot(data=comparacao.reset_index(), y='index', x='num_anomalies', ax=ax)
    ax.set_title("Número de Anomalias Detectadas por Cada Método")
    st.pyplot(fig_metrica)

if __name__ == "__main__":
    main()
