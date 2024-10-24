#%%
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from scipy.stats import pearsonr;
import matplotlib.gridspec as gridspec
from matplotlib_venn import venn2, venn3

################### Iniciando Mineração de Dados #############
df = pd.read_pickle('dados/processados/eda.pkl')

# Excluindo colunas do tipo data
df = df.select_dtypes(exclude=['datetime64[ns]'])

# Imputar valor 1 para os valores ausentes na coluna duracaoProposta
df['duracaoProposta'].fillna(1, inplace=True)

# Converter booleanos em int
df = df.astype(int)

#normalização dos dados. 
scaler = StandardScaler()
df_normalizados = scaler.fit_transform(df)
df_normalizados = pd.DataFrame(df_normalizados, columns=df.columns)

# Configurar o estilo do Seaborn
sns.set_theme(style="whitegrid", palette="deep")
sns.set()

#treinamento de modelos

def apply_pca(df_normalizados, n_components=2):
    """
    Aplica PCA aos dados normalizados
    """
    pca = PCA(n_components=n_components)
    dados_pca = pca.fit_transform(df_normalizados)
    return dados_pca, pca

def create_pca_dataframe(dados_pca, df, column):
    """
    Cria um DataFrame com os resultados do PCA e informações de anomalia
    """
    return pd.DataFrame({
        'PC1': dados_pca[:, 0],
        'PC2': dados_pca[:, 1],
        'Tipo': np.where(df[column], 'Anomalia', 'Normal')
    })

def detect_anomalies_kmeans(df, df_normalizados, n_clusters=2, percentil=95, random_state=42):
    """
    Detecta anomalias usando K-Means
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster_kmeans'] = kmeans.fit_predict(df_normalizados)
    
    # Cálculo das distâncias aos centróides
    centroides = kmeans.cluster_centers_
    distancias = np.linalg.norm(df_normalizados - centroides[df['cluster_kmeans']], axis=1)
    df['distancia_do_centroide'] = distancias
    
    # Identificação de anomalias
    limite = np.percentile(df['distancia_do_centroide'], percentil)
    df['anomalia_kmeans'] = df['distancia_do_centroide'] > limite
    
    return df, kmeans

def detect_anomalies_dbscan(df, df_normalizados, eps=0.5, min_samples=5):
    """
    Detecta anomalias usando DBSCAN
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster_dbscan'] = dbscan.fit_predict(df_normalizados)
    
    # No DBSCAN, -1 indica ruído (anomalias)
    df['anomalia_dbscan'] = df['cluster_dbscan'] == -1
    
    return df, dbscan

def detect_anomalies_lof(df, df_normalizados, n_neighbors=20, contamination=0.1):
    """
    Detecta anomalias usando Local Outlier Factor (LOF)
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    df['anomalia_lof'] = lof.fit_predict(df_normalizados) == -1
    df['lof_score'] = -lof.negative_outlier_factor_
    
    return df

def detect_anomalies_iforest(df, df_normalizados, contamination=0.1, random_state=42):
    """
    Detecta anomalias usando Isolation Forest
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    df['anomalia_iforest'] = iso_forest.fit_predict(df_normalizados) == -1
    
    return df, iso_forest

def detect_all_anomalies(df, df_normalizados, params=None):
    """
    Aplica todos os métodos de detecção de anomalias
    """
    if params is None:
        params = {
            'kmeans': {'n_clusters': 2, 'percentil': 95},
            'dbscan': {'eps': 0.5, 'min_samples': 5},
            'lof': {'n_neighbors': 20, 'contamination': 0.1},
            'iforest': {'contamination': 0.1}
        }
    
    # Aplicar cada método separadamente
    df_temp = df.copy()
    
    # K-Means
    n_clusters = params['kmeans'].get('n_clusters', 2)
    percentil = params['kmeans'].get('percentil', 95)
    df_temp, kmeans = detect_anomalies_kmeans(df_temp, df_normalizados, 
                                            n_clusters=n_clusters, 
                                            percentil=percentil)
    
    # DBSCAN
    eps = params['dbscan'].get('eps', 0.5)
    min_samples = params['dbscan'].get('min_samples', 5)
    df_temp, dbscan = detect_anomalies_dbscan(df_temp, df_normalizados,
                                            eps=eps,
                                            min_samples=min_samples)
    
    # LOF
    n_neighbors = params['lof'].get('n_neighbors', 20)
    lof_contamination = params['lof'].get('contamination', 0.1)
    df_temp = detect_anomalies_lof(df_temp, df_normalizados,
                                 n_neighbors=n_neighbors,
                                 contamination=lof_contamination)
    
    # Isolation Forest
    iforest_contamination = params['iforest'].get('contamination', 0.1)
    df_temp, iforest = detect_anomalies_iforest(df_temp, df_normalizados,
                                              contamination=iforest_contamination)
    
    # Criar coluna de consenso
    anomaly_columns = ['anomalia_kmeans', 'anomalia_dbscan', 
                      'anomalia_lof', 'anomalia_iforest']
    df_temp['anomalia_consenso'] = df_temp[anomaly_columns].sum(axis=1)
    
    return df_temp, {'kmeans': kmeans, 'dbscan': dbscan, 'iforest': iforest}

def plot_anomalies_comparison(df, df_normalizados):
    """
    Plota comparação das anomalias detectadas por diferentes métodos
    
    Parâmetros:
    df: DataFrame com as colunas de anomalias
    df_normalizados: array numpy com os dados normalizados
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('Comparação de Métodos de Detecção de Anomalias', size=16)
    
    methods = {
        'K-Means': 'anomalia_kmeans',
        'DBSCAN': 'anomalia_dbscan',
        'LOF': 'anomalia_lof',
        'Isolation Forest': 'anomalia_iforest'
    }
    
    for (title, column), ax in zip(methods.items(), axes.ravel()):
        # Converter a máscara booleana para array numpy
        mask = df[column].values
        
        # Plotar pontos normais (onde mask é False)
        normal_mask = ~mask
        ax.scatter(df_normalizados[normal_mask, 0], 
                  df_normalizados[normal_mask, 1],
                  c='blue', alpha=0.5, label='Normal')
        
        # Plotar anomalias (onde mask é True)
        anomaly_mask = mask
        ax.scatter(df_normalizados[anomaly_mask, 0],
                  df_normalizados[anomaly_mask, 1],
                  c='red', alpha=0.7, label='Anomalia')
        
        ax.set_title(f'Anomalias - {title}')
        ax.set_xlabel('Valor Total Estimado (Normalizado)')
        ax.set_ylabel('Duração da Proposta (Normalizada)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('images/anomalies_comparison.png')  # Salva automaticamente ao plotar
    plt.show()

def plot_consensus_anomalies(df, df_normalizados):
    """
    Plota anomalias baseadas no consenso entre os métodos
    
    Parâmetros:
    df: DataFrame com a coluna 'anomalia_consenso'
    df_normalizados: array numpy com os dados normalizados
    """
    plt.figure(figsize=(12, 8))
    
    # Usar diretamente as coordenadas x e y dos dados normalizados
    scatter = plt.scatter(df_normalizados[:, 0], df_normalizados[:, 1],
                         c=df['anomalia_consenso'].values, cmap='YlOrRd',
                         alpha=0.6, s=100)
    
    plt.colorbar(scatter, label='Número de métodos que identificaram como anomalia')
    plt.xlabel('Valor Total Estimado (Normalizado)')
    plt.ylabel('Duração da Proposta (Normalizada)')
    plt.title('Consenso entre Métodos de Detecção de Anomalias')
    plt.grid(True)
    plt.savefig('images/concensus_anomalies.png')  # Salva automaticamente ao plotar
    plt.show()

def print_anomaly_summary(df):
    """
    Imprime um resumo das anomalias detectadas por cada método
    """
    methods = {
        'K-Means': 'anomalia_kmeans',
        'DBSCAN': 'anomalia_dbscan',
        'LOF': 'anomalia_lof',
        'Isolation Forest': 'anomalia_iforest'
    }
    
    print("\nResumo das Anomalias Detectadas:")
    print("-" * 50)
    for method_name, column in methods.items():
        n_anomalies = df[column].sum()
        percent = (n_anomalies / len(df)) * 100
        print(f"{method_name}: {n_anomalies} anomalias ({percent:.2f}%)")
    
    print("\nConsenso entre métodos:")
    print(df['anomalia_consenso'].value_counts().sort_index()
          .apply(lambda x: f"{x} pontos detectados por {x} métodos"))
    
def evaluate_clustering_model(df, df_normalizados, model_name, labels):
    """
    Avalia o modelo de clustering usando diferentes métricas
    """
    # Ignorar rótulos de ruído (-1) para métricas que não os suportam
    valid_points = labels != -1 if model_name == 'DBSCAN' else np.ones_like(labels, dtype=bool)
    
    metrics = {}
    
    try:
        metrics['silhouette'] = silhouette_score(
            df_normalizados[valid_points], 
            labels[valid_points]
        )
    except:
        metrics['silhouette'] = np.nan
        
    try:
        metrics['calinski'] = calinski_harabasz_score(
            df_normalizados[valid_points], 
            labels[valid_points]
        )
    except:
        metrics['calinski'] = np.nan
        
    try:
        metrics['davies'] = davies_bouldin_score(
            df_normalizados[valid_points], 
            labels[valid_points]
        )
    except:
        metrics['davies'] = np.nan
    
    # Calcular estatísticas das anomalias
    n_anomalies = np.sum(labels == -1) if model_name == 'DBSCAN' else np.sum(labels == 1)
    metrics['anomaly_ratio'] = n_anomalies / len(labels)
    
    return metrics

def detect_all_anomalies_with_evaluation(df, df_normalizados, params=None):
    """
    Aplica todos os métodos de detecção de anomalias e avalia seus desempenhos
    """
    if params is None:
        params = {
            'kmeans': {'n_clusters': 2, 'percentil': 95},
            'dbscan': {'eps': 0.5, 'min_samples': 5},
            'lof': {'n_neighbors': 20, 'contamination': 0.1},
            'iforest': {'contamination': 0.1}
        }
    
    df_temp = df.copy()
    models = {}
    evaluations = {}
    
    # K-Means
    df_temp, kmeans = detect_anomalies_kmeans(df_temp, df_normalizados, 
                                            n_clusters=params['kmeans']['n_clusters'], 
                                            percentil=params['kmeans']['percentil'])
    models['kmeans'] = kmeans
    evaluations['K-Means'] = evaluate_clustering_model(
        df, df_normalizados, 'K-Means', df_temp['cluster_kmeans']
    )
    
    # DBSCAN
    df_temp, dbscan = detect_anomalies_dbscan(df_temp, df_normalizados,
                                            eps=params['dbscan']['eps'],
                                            min_samples=params['dbscan']['min_samples'])
    models['dbscan'] = dbscan
    evaluations['DBSCAN'] = evaluate_clustering_model(
        df, df_normalizados, 'DBSCAN', df_temp['cluster_dbscan']
    )
    
    # LOF
    df_temp = detect_anomalies_lof(df_temp, df_normalizados,
                                 n_neighbors=params['lof']['n_neighbors'],
                                 contamination=params['lof']['contamination'])
    evaluations['LOF'] = {
        'anomaly_ratio': df_temp['anomalia_lof'].mean(),
        'avg_score': df_temp['lof_score'].mean()
    }
    
    # Isolation Forest
    df_temp, iforest = detect_anomalies_iforest(df_temp, df_normalizados,
                                              contamination=params['iforest']['contamination'])
    models['iforest'] = iforest
    evaluations['Isolation Forest'] = {
        'anomaly_ratio': df_temp['anomalia_iforest'].mean()
    }
    
    # Criar coluna de consenso
    anomaly_columns = ['anomalia_kmeans', 'anomalia_dbscan', 
                      'anomalia_lof', 'anomalia_iforest']
    df_temp['anomalia_consenso'] = df_temp[anomaly_columns].astype(int).sum(axis=1)
    
    return df_temp, models, evaluations

def plot_best_model(df, df_normalizados, model_name, anomaly_column):
    """
    Cria visualizações detalhadas para o melhor modelo
    """
    plt.figure(figsize=(15, 10))
    
    # Plot principal
    plt.subplot(2, 2, 1)
    mask = df[anomaly_column].values
    plt.scatter(df_normalizados[~mask, 0], df_normalizados[~mask, 1],
               c='blue', alpha=0.5, label='Normal')
    plt.scatter(df_normalizados[mask, 0], df_normalizados[mask, 1],
               c='red', alpha=0.7, label='Anomalia')
    plt.title(f'Anomalias Detectadas - {model_name}')
    plt.xlabel('Valor Total Estimado (Normalizado)')
    plt.ylabel('Duração da Proposta (Normalizada)')
    plt.legend()
    plt.grid(True)

    
    # Distribuição das características para pontos normais vs anomalias
    plt.subplot(2, 2, 2)
    plt.hist(df.loc[~mask, 'valorTotalEstimado'], bins=30, alpha=0.5, 
             label='Normal', density=True)
    plt.hist(df.loc[mask, 'valorTotalEstimado'], bins=30, alpha=0.5,
             label='Anomalia', density=True)
    plt.title('Distribuição do Valor Total Estimado')
    plt.xlabel('Valor')
    plt.ylabel('Densidade')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.hist(df.loc[~mask, 'duracaoProposta'], bins=30, alpha=0.5,
             label='Normal', density=True)
    plt.hist(df.loc[mask, 'duracaoProposta'], bins=30, alpha=0.5,
             label='Anomalia', density=True)
    plt.title('Distribuição da Duração da Proposta')
    plt.xlabel('Duração')
    plt.ylabel('Densidade')
    plt.legend()
    
    # Box plot das características
    plt.subplot(2, 2, 4)
    data_to_plot = [
        df.loc[~mask, 'valorTotalEstimado'],
        df.loc[mask, 'valorTotalEstimado'],
        df.loc[~mask, 'duracaoProposta'],
        df.loc[mask, 'duracaoProposta']
    ]
    labels = ['Normal\nValor', 'Anomalia\nValor', 
              'Normal\nDuração', 'Anomalia\nDuração']
    plt.boxplot(data_to_plot, labels=labels)
    plt.title('Comparação das Distribuições')
    plt.xticks(rotation=45)    
    plt.tight_layout()
    plt.savefig('images/best_model.png')  # Salva automaticamente ao plotar
    plt.show()

def print_model_evaluation(evaluations):
    """
    Imprime uma análise detalhada das métricas de avaliação
    """
    print("\nAvaliação dos Modelos:")
    print("-" * 50)
    
    # Criar um DataFrame para melhor visualização
    metrics_df = pd.DataFrame(evaluations).T
    
    print("\nMétricas de avaliação:")
    print(metrics_df)
    
    # Identificar o melhor modelo baseado nas métricas disponíveis
    best_model = None
    best_score = -np.inf
    
    for model, metrics in evaluations.items():
        if 'silhouette' in metrics and not np.isnan(metrics['silhouette']):
            if metrics['silhouette'] > best_score:
                best_score = metrics['silhouette']
                best_model = model
    
    if best_model:
        print(f"\nModelo recomendado: {best_model}")
        print(f"Métricas do modelo recomendado:")
        for metric, value in evaluations[best_model].items():
            print(f"- {metric}: {value:.4f}")
    
    return best_model

def plot_anomalies_comparison_pca(df, df_normalizados):
    """
    Plota comparação das anomalias detectadas por diferentes métodos usando PCA com Seaborn
    """
    dados_pca, pca = apply_pca(df_normalizados)
    var_ratio = pca.explained_variance_ratio_
    
    fig = plt.figure(figsize=(20, 20))
    
    methods = {
        'K-Means': 'anomalia_kmeans',
        'DBSCAN': 'anomalia_dbscan',
        'LOF': 'anomalia_lof',
        'Isolation Forest': 'anomalia_iforest'
    }
    
    for idx, (title, column) in enumerate(methods.items(), 1):
        plt.subplot(2, 2, idx)
        
        # Criar DataFrame para Seaborn
        plot_df = create_pca_dataframe(dados_pca, df, column)
        
        # Criar scatter plot com Seaborn
        sns.scatterplot(
            data=plot_df,
            x='PC1',
            y='PC2',
            hue='Tipo',
            style='Tipo',
            palette={'Normal': 'blue', 'Anomalia': 'red'},
            alpha=0.6,
            s=100
        )
        
        plt.title(f'Anomalias - {title}\nVariância explicada: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}')
        plt.xlabel('Primeira Componente Principal')
        plt.ylabel('Segunda Componente Principal')
    
    plt.tight_layout()
    plt.savefig('images/anomalies_comparison_pca.png')  # Salva automaticamente ao plotar
    plt.show()

def plot_consensus_anomalies_pca(df, df_normalizados):
    """
    Plota anomalias baseadas no consenso entre os métodos usando PCA com Seaborn
    """
    dados_pca, pca = apply_pca(df_normalizados)
    var_ratio = pca.explained_variance_ratio_
    
    plt.figure(figsize=(12, 8))
    
    # Criar DataFrame para Seaborn
    plot_df = pd.DataFrame({
        'PC1': dados_pca[:, 0],
        'PC2': dados_pca[:, 1],
        'Consenso': df['anomalia_consenso']
    })
    
    # Usar Seaborn para criar um scatter plot com gradiente de cores
    sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='Consenso',
        palette='YlOrRd',
        size='Consenso',
        sizes=(100, 400),
        legend='full'
    )
    
    plt.title(f'Consenso entre Métodos de Detecção de Anomalias (PCA)\n' +
              f'Variância explicada: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}')
    plt.xlabel('Primeira Componente Principal')
    plt.ylabel('Segunda Componente Principal')
    plt.savefig('images/concensus_anomalies_pca.png')  # Salva automaticamente ao plotar
    plt.show()

def plot_best_model_pca(df, df_normalizados, model_name, anomaly_column):
    """
    Cria visualizações detalhadas para o melhor modelo usando PCA e Seaborn
    """
    dados_pca, pca = apply_pca(df_normalizados)
    var_ratio = pca.explained_variance_ratio_
    
    # Criar DataFrame com resultados do PCA
    plot_df = create_pca_dataframe(dados_pca, df, anomaly_column)
    plot_df['Valor Total'] = df['valorTotalEstimado']
    plot_df['Duração'] = df['duracaoProposta']
    
    # Configurar o grid de subplots
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 3, figure=fig)
    
    # 1. Scatter plot principal com PCA
    ax1 = fig.add_subplot(gs[0, :2])
    sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='Tipo',
        style='Tipo',
        palette={'Normal': 'blue', 'Anomalia': 'red'},
        alpha=0.6,
        s=100,
        ax=ax1
    )
    ax1.set_title(f'Anomalias Detectadas - {model_name}\n' +
                  f'Variância explicada: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}')
    
    # 2. Scree plot
    ax2 = fig.add_subplot(gs[0, 2])
    sns.lineplot(
        x=range(1, len(var_ratio) + 1),
        y=np.cumsum(var_ratio),
        marker='o',
        ax=ax2
    )
    ax2.set_title('Scree Plot')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Variância Explicada Acumulada')
    
    # 3. Distribuições das features originais
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxenplot(
        data=plot_df,
        x='Tipo',
        y='Valor Total',
        ax=ax3
    )
    ax3.set_title('Distribuição do Valor Total por Tipo')
    
    ax4 = fig.add_subplot(gs[1, 1])
    sns.boxenplot(
        data=plot_df,
        x='Tipo',
        y='Duração',
        ax=ax4
    )
    ax4.set_title('Distribuição da Duração por Tipo')
    
    # 4. Joint plot das componentes principais
    ax5 = fig.add_subplot(gs[1, 2])
    sns.kdeplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='Tipo',
        ax=ax5,
        fill=True,
        alpha=0.5
    )
    ax5.set_title('Densidade das Componentes Principais')
    plt.tight_layout()
    plt.savefig('images/best_model_pca_densidade.png')  # Salva automaticamente ao plotar
    plt.show()
    
    # Plot adicional: Pair plot para análise multivariada
    pair_df = plot_df[['PC1', 'PC2', 'Valor Total', 'Duração', 'Tipo']]
    sns.pairplot(
        pair_df,
        hue='Tipo',
        palette={'Normal': 'blue', 'Anomalia': 'red'},
        diag_kind='kde'
    )
    plt.savefig('images/best_model_pair_plot.png')  # Salva automaticamente ao plotar
    plt.show()

def plot_correlation_analysis(df, df_normalizados):
    """
    Análise de correlação entre as features originais e as componentes principais
    """
    dados_pca, pca = apply_pca(df_normalizados)
    
    # Criar DataFrame com features originais e componentes principais
    corr_df = pd.DataFrame({
        'Valor Total': df['valorTotalEstimado'],
        'Duração': df['duracaoProposta'],
        'PC1': dados_pca[:, 0],
        'PC2': dados_pca[:, 1]
    })
    
    # Calcular matriz de correlação
    corr_matrix = corr_df.corr()
    
    # Plotar mapa de calor das correlações
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Correlação entre Features Originais e Componentes Principais')
    plt.savefig('images/correlation_analysis.png')  # Salva automaticamente ao plotagem
    plt.show()

def run_anomaly_detection_analysis(df, df_normalizados, params=None):
    """
    Executa a análise completa de detecção de anomalias
    """
    # Detectar anomalias e avaliar modelos
    df_results, models, evaluations = detect_all_anomalies_with_evaluation(
        df, df_normalizados, params
    )
    
    # Imprimir avaliação e identificar melhor modelo
    best_model = print_model_evaluation(evaluations)
    
    # Mapear nome do modelo para coluna de anomalia correspondente
    anomaly_columns = {
        'K-Means': 'anomalia_kmeans',
        'DBSCAN': 'anomalia_dbscan',
        'LOF': 'anomalia_lof',
        'Isolation Forest': 'anomalia_iforest'
    }
    
    if best_model and best_model in anomaly_columns:
        # Plotar visualizações detalhadas para o melhor modelo
        plot_best_model(
            df_results, 
            df_normalizados, 
            best_model,
            anomaly_columns[best_model]
        )
    
    # Visualizar resultados usando df_results
    plot_anomalies_comparison(df_results, df_normalizados)
    plot_consensus_anomalies(df_results, df_normalizados)
    print_anomaly_summary(df_results)
    
    return df_results, models, evaluations, best_model

def run_anomaly_detection_analysis_pca(df, df_normalizados, params=None):
    """
    Executa a análise completa de detecção de anomalias com visualizações aprimoradas
    """
    # Detectar anomalias e avaliar modelos
    df_results, models, evaluations = detect_all_anomalies_with_evaluation(
        df, df_normalizados, params
    )
    
    # Imprimir avaliação e identificar melhor modelo
    best_model = print_model_evaluation(evaluations)
    
    # Mapear nome do modelo para coluna de anomalia
    anomaly_columns = {
        'K-Means': 'anomalia_kmeans',
        'DBSCAN': 'anomalia_dbscan',
        'LOF': 'anomalia_lof',
        'Isolation Forest': 'anomalia_iforest'
    }
    
    if best_model and best_model in anomaly_columns:
        # Plotar visualizações detalhadas para o melhor modelo
        plot_best_model_pca(
            df_results, 
            df_normalizados, 
            best_model,
            anomaly_columns[best_model]
        )
    
    # Visualizar resultados
    plot_anomalies_comparison_pca(df_results, df_normalizados)
    plot_consensus_anomalies_pca(df_results, df_normalizados)
    plot_correlation_analysis(df_results, df_normalizados)
    print_anomaly_summary(df_results)
    
    return df_results, models, evaluations, best_model

# Configurar parâmetros
params = {
    'kmeans': {'n_clusters': 2, 'percentil': 95},
    'dbscan': {'eps': 0.5, 'min_samples': 5},
    'lof': {'n_neighbors': 20, 'contamination': 0.1},
    'iforest': {'contamination': 0.1}
}

# Normalizar os dados (se ainda não estiver feito)
if isinstance(df_normalizados, pd.DataFrame):
    df_normalizados = df_normalizados.values


df_normalizados = scaler.fit_transform(df)

# Executar análise completa
df, models, evaluations, best_model = run_anomaly_detection_analysis(
    df, df_normalizados, params
)

df, models, evaluations, best_model = run_anomaly_detection_analysis_pca(df, df_normalizados, params)

#
# Visualizar resultados
plot_anomalies_comparison(df, df_normalizados)
plot_consensus_anomalies(df, df_normalizados)
print_anomaly_summary(df)

############################### Rede Neural ###################################

class DeepAnomalyDetector:
    def __init__(self, encoding_dim=8, hidden_layers=[32, 16]):
        """
        Inicializa o detector de anomalias baseado em Autoencoder
        
        Parâmetros:
        encoding_dim: dimensão da camada do meio (encoded representation)
        hidden_layers: lista com número de neurônios nas camadas intermediárias
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.autoencoder = None
        self.encoder = None
        self.history = None
        self.threshold = None
        self.scaler = StandardScaler()
        
    def create_autoencoder(self, input_dim):
        """
        Cria a arquitetura do autoencoder
        """
        # Input
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in self.hidden_layers:
            encoded = Dense(units, activation='relu')(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(0.2)(encoded)
            
        # Bottleneck
        bottleneck = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = bottleneck
        for units in reversed(self.hidden_layers):
            decoded = Dense(units, activation='relu')(decoded)
            decoded = BatchNormalization()(decoded)
            decoded = Dropout(0.2)(decoded)
            
        # Output
        output_layer = Dense(input_dim, activation='linear')(decoded)
        
        # Create models
        self.autoencoder = Model(input_layer, output_layer)
        self.encoder = Model(input_layer, bottleneck)
        
        # Compile
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
    def fit(self, X, validation_split=0.1, epochs=100, batch_size=32):
        """
        Treina o autoencoder
        """
        # Normaliza os dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Criar o modelo se ainda não existir
        if self.autoencoder is None:
            self.create_autoencoder(X.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Treinar o modelo
        self.history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calcular o threshold baseado nos erros de reconstrução
        reconstructed = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95º percentil
        
        return self
    
    def predict(self, X):
        """
        Detecta anomalias nos dados
        Retorna: array boolean (True para anomalias)
        """
        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        return mse > self.threshold
    
    def get_anomaly_scores(self, X):
        """
        Retorna os scores de anomalia (erro de reconstrução)
        """
        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled)
        return np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    
    def plot_training_history(self):
        """
        Plota o histórico de treinamento
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_detection(self, X, anomalies=None):
        """
        Visualiza as anomalias detectadas
        """
        scores = self.get_anomaly_scores(X)
        predictions = scores > self.threshold
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Scatter plot dos dados com anomalias destacadas
        plt.subplot(2, 2, 1)
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], 
                   c=predictions, cmap='coolwarm',
                   alpha=0.6)
        plt.title('Anomalias Detectadas')
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        
        # Plot 2: Distribuição dos scores de anomalia
        plt.subplot(2, 2, 2)
        sns.histplot(scores, bins=50)
        plt.axvline(self.threshold, color='r', linestyle='--',
                   label=f'Threshold ({self.threshold:.3f})')
        plt.title('Distribuição dos Scores de Anomalia')
        plt.xlabel('Score de Anomalia')
        plt.ylabel('Contagem')
        plt.legend()
        
        # Plot 3: Comparação com ground truth (se disponível)
        if anomalies is not None:
            plt.subplot(2, 2, 3)
            conf_matrix = tf.math.confusion_matrix(anomalies, predictions)
            sns.heatmap(conf_matrix, annot=True, fmt='d',
                       xticklabels=['Normal', 'Anomalia'],
                       yticklabels=['Normal', 'Anomalia'])
            plt.title('Matriz de Confusão')
            
            # Métricas
            accuracy = np.mean(predictions == anomalies)
            precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1])
            recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
            f1 = 2 * (precision * recall) / (precision + recall)
            
            plt.subplot(2, 2, 4)
            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }
            plt.bar(metrics.keys(), metrics.values())
            plt.title('Métricas de Avaliação')
            plt.ylim(0, 1)
            
        plt.tight_layout()
        plt.savefig('images/anomaly_detection.png')  # Salva automaticamente ao plotar

        plt.show()

def compare_with_traditional(df, deep_predictions, traditional_predictions):
    """
    Compara as predições do modelo deep learning com as do método tradicional
    """
    plt.figure(figsize=(15, 5))
    
    # Deep Learning
    plt.subplot(1, 2, 1)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
               c=deep_predictions, cmap='coolwarm', alpha=0.6)
    plt.title('Anomalias - Deep Learning')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    
    # Método Tradicional
    plt.subplot(1, 2, 2)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
               c=traditional_predictions, cmap='coolwarm', alpha=0.6)
    plt.title('Anomalias - Método Tradicional')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    
    plt.tight_layout()
    plt.savefig('images/compare_with_traditional.png')  # Salva automaticamente ao plotar
    plt.show()
    
    # Calcular concordância entre os métodos
    agreement = np.mean(deep_predictions == traditional_predictions)
    print(f"\nConcordância entre os métodos: {agreement:.2%}")
    
    # Análise detalhada das diferenças
    diff_mask = deep_predictions != traditional_predictions
    if np.any(diff_mask):
        print("\nAnálise dos pontos com classificação diferente:")
        diff_points = df[diff_mask]
        print(f"Número de pontos com classificação diferente: {len(diff_points)}")
        print("\nEstatísticas dos pontos divergentes:")
        print(diff_points.describe())

# Exemplo de uso:
def run_deep_anomaly_detection(df, traditional_anomalies=None):
    """
    Executa a análise completa usando deep learning
    """
    # Criar e treinar o modelo
    detector = DeepAnomalyDetector(
        encoding_dim=8,
        hidden_layers=[32, 16]
    )
    
    # Treinar o modelo
    detector.fit(df, epochs=100, batch_size=32)
    
    # Plotar histórico de treinamento
    detector.plot_training_history()
    
    # Detectar anomalias
    anomalies = detector.predict(df)

    
    # Visualizar resultados
    detector.plot_anomaly_detection(df, traditional_anomalies)
    
    # Se houver resultados do método tradicional, fazer comparação
    if traditional_anomalies is not None:
        compare_with_traditional(df, anomalies, traditional_anomalies)
    
    return detector, anomalies


# Obter as anomalias do melhor modelo tradicional
traditional_anomalies = df[f'anomalia_{best_model.lower()}']

# Executar o detector baseado em deep learning
detector, deep_anomalies = run_deep_anomaly_detection(df, traditional_anomalies)

####################### Analise Final #########################
class TraditionalDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def fit_predict_kmeans(self, X, n_clusters=2, percentile=95):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Calculate distances to centroids
        distances = kmeans.transform(X).min(axis=1)
        threshold = np.percentile(distances, percentile)
        anomalies = distances > threshold
        
        self.models['KMeans'] = kmeans
        return pd.Series(anomalies, index=X.index)
    
    def fit_predict_dbscan(self, X, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        anomalies = labels == -1
        
        self.models['DBSCAN'] = dbscan
        return pd.Series(anomalies, index=X.index)
    
    def fit_predict_lof(self, X, n_neighbors=20, contamination=0.1):
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, 
                                contamination=contamination,
                                novelty=True)
        lof.fit(X)
        predictions = lof.predict(X)
        anomalies = predictions == -1
        
        self.models['LOF'] = lof
        return pd.Series(anomalies, index=X.index)
    
    def fit_predict_iforest(self, X, contamination=0.1):
        iforest = IsolationForest(contamination=contamination, 
                                 random_state=42)
        predictions = iforest.fit_predict(X)
        anomalies = predictions == -1
        
        self.models['IForest'] = iforest
        return pd.Series(anomalies, index=X.index)
    
    def fit_predict_all(self, X, params=None):
        if params is None:
            params = {
                'kmeans': {'n_clusters': 2, 'percentile': 95},
                'dbscan': {'eps': 0.5, 'min_samples': 5},
                'lof': {'n_neighbors': 20, 'contamination': 0.1},
                'iforest': {'contamination': 0.1}
            }
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        results = pd.DataFrame({
            'KMeans': self.fit_predict_kmeans(X_scaled, **params['kmeans']),
            'DBSCAN': self.fit_predict_dbscan(X_scaled, **params['dbscan']),
            'LOF': self.fit_predict_lof(X_scaled, **params['lof']),
            'IForest': self.fit_predict_iforest(X_scaled, **params['iforest'])
        }, index=X.index)
        
        return results

class DeepAnomalyDetector:
    def __init__(self, encoding_dim=8, hidden_layers=[32, 16]):
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.autoencoder = None
        self.encoder = None
        self.history = None
        self.threshold = None
        self.scaler = StandardScaler()
        
    def create_autoencoder(self, input_dim):
        """
        Create the autoencoder architecture with the specified dimensions.
        
        Args:
            input_dim (int): Number of input features
        """
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in self.hidden_layers:
            encoded = Dense(units, activation='relu')(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(0.2)(encoded)
            
        # Bottleneck layer
        bottleneck = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder (mirror of encoder)
        decoded = bottleneck
        for units in reversed(self.hidden_layers):
            decoded = Dense(units, activation='relu')(decoded)
            decoded = BatchNormalization()(decoded)
            decoded = Dropout(0.2)(decoded)
            
        # Output layer
        output_layer = Dense(input_dim, activation='linear')(decoded)
        
        # Create models
        self.autoencoder = Model(input_layer, output_layer)
        self.encoder = Model(input_layer, bottleneck)
        
        # Compile autoencoder
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
    def fit(self, X, validation_split=0.1, epochs=100, batch_size=32):
        """
        Fit the autoencoder to the data.
        
        Args:
            X: Input data
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            self
        """
        X_scaled = self.scaler.fit_transform(X)
        
        if self.autoencoder is None:
            self.create_autoencoder(X.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculate reconstruction error threshold
        reconstructed = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        self.threshold = np.percentile(mse, 95)
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies in new data.
        
        Args:
            X: Input data
            
        Returns:
            Series of boolean values (True for anomalies)
        """
        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        return pd.Series(mse > self.threshold, index=X.index)
    
    def get_reconstruction_error(self, X):
        """
        Get reconstruction error for each sample.
        
        Args:
            X: Input data
            
        Returns:
            Array of reconstruction errors
        """
        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled)
        return np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

class AnomalyAnalyzer:
    def __init__(self, data):
        self.data = data
        self.results = None
        self.pca = None
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        scaled_data = self.scaler.fit_transform(self.data)
        self.pca = PCA(n_components=2)
        self.pca_data = pd.DataFrame(
            self.pca.fit_transform(scaled_data),
            columns=['PC1', 'PC2'],
            index=self.data.index
        )
        
    def analyze_consensus(self, method_results):
        # Ensure method_results has the same index as the data
        self.results = pd.DataFrame(method_results, index=self.data.index)
        self.results['n_detections'] = self.results.sum(axis=1)
        
        consensus_levels = {
            'Total': pd.Series(self.results['n_detections'] == len(method_results.columns), 
                             index=self.data.index),
            'Forte': pd.Series(self.results['n_detections'] >= len(method_results.columns) - 1,
                              index=self.data.index),
            'Moderada': pd.Series(self.results['n_detections'] >= len(method_results.columns) - 2,
                                index=self.data.index),
            'Fraca': pd.Series(self.results['n_detections'] >= 1,
                            index=self.data.index)
        }
        
        self.plot_consensus_analysis(consensus_levels)
        self.print_consensus_statistics(consensus_levels)
        
        return consensus_levels
        
    def plot_consensus_analysis(self, consensus_levels):
        plt.figure(figsize=(40, 30))
        
        # Definir o tamanho da fonte globalmente
        plt.rcParams.update({'font.size': 24})  # Define o tamanho da fonte para todo o gráfico
        
        # PCA scatter plot
        plt.subplot(2, 2, 1)
        
        # Plot all points first
        plt.scatter(self.pca_data['PC1'], self.pca_data['PC2'], 
                c='lightgray', alpha=0.3, label='Normal')
        
        # Plot anomalies with different colors
        colors = ['red', 'orange', 'yellow', 'green']
        for (level_name, mask), color in zip(consensus_levels.items(), colors):
            anomaly_points = self.pca_data[mask]
            if not anomaly_points.empty:
                plt.scatter(anomaly_points['PC1'], 
                        anomaly_points['PC2'],
                        c=color, 
                        alpha=0.6, 
                        s=300,
                        label=f'{level_name} Consensus')
        
        plt.title('Análises dos Consensos  (PCA)', fontsize=28)  # Ajustar o tamanho do título
        plt.xlabel('Primeira Componente Principal', fontsize=24)  # Ajustar o tamanho do rótulo do eixo X
        plt.ylabel('Segunda Componente Principal', fontsize=24)  # Ajustar o tamanho do rótulo do eixo Y
        plt.legend(fontsize=20)  # Ajustar o tamanho da legenda
        
        # Detection distribution
        plt.subplot(2, 2, 2)
        self.results['n_detections'].hist(bins=len(self.results.columns)-1)
        plt.title('Detecção de Distribuição', fontsize=28)
        plt.xlabel('Número de Métodos', fontsize=24)
        plt.ylabel('Frequência', fontsize=24)
        
        # Consensus levels
        plt.subplot(2, 2, 3)
        counts = [mask.sum() for mask in consensus_levels.values()]
        plt.bar(consensus_levels.keys(), counts)
        plt.title('Anomalias por Nível de Consenso', fontsize=28)
        plt.xticks(rotation=45, fontsize=22)  # Ajustar o tamanho das marcas no eixo X
        
        # Feature importance
        plt.subplot(2, 2, 4)
        importance = pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f'PC{i+1}' for i in range(len(self.pca.explained_variance_ratio_))]
        )
        importance.plot(kind='bar')
        plt.title('PCA - Razão de Variância Explicada', fontsize=28)
        
        plt.tight_layout()
        plt.savefig("images/consensus_analisys.png")
        plt.show()

        
    def print_consensus_statistics(self, consensus_levels):
        print("\nConsensus Statistics:")
        for level_name, mask in consensus_levels.items():
            n_anomalies = mask.sum()
            percentage = (n_anomalies / len(self.data)) * 100
            print(f"\n{level_name} Consensus:")
            print(f"Number of anomalies: {n_anomalies}")
            print(f"Percentage of total: {percentage:.2f}%")
            
            if n_anomalies > 0:
                anomaly_data = self.data[mask]
                print("\nDescriptive statistics for anomalies:")
                print(anomaly_data.describe().round(2))

def run_complete_analysis(data):
    """
    Run complete anomaly detection analysis using all methods.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary containing all detectors and results
    """
    # 1. Traditional Methods
    trad_detector = TraditionalDetector()
    traditional_results = trad_detector.fit_predict_all(data)
    
    # 2. Deep Learning
    deep_detector = DeepAnomalyDetector()
    deep_detector.fit(data)
    deep_predictions = deep_detector.predict(data)
    
    # 3. Combine all results
    all_results = traditional_results.copy()
    all_results['DeepLearning'] = deep_predictions
    
    # 4. Analyze results
    analyzer = AnomalyAnalyzer(data)
    analyzer.prepare_data()
    consensus_levels = analyzer.analyze_consensus(all_results)
    
    return {
        'traditional_detector': trad_detector,
        'deep_detector': deep_detector,
        'analyzer': analyzer,
        'consensus_levels': consensus_levels,
        'all_results': all_results
    }

# Exemplo de uso:
# Carregar seus dados
data = df # Seus dados aqui

# Executar análise completa
results = run_complete_analysis(data)

# Acessar resultados
traditional_detector = results['traditional_detector']
deep_detector = results['deep_detector']
analyzer = results['analyzer']
consensus_levels = results['consensus_levels']
all_results = results['all_results']

# Ver estatísticas dos níveis de consenso
print("\nEstatísticas por nível de consenso:")
for level_name, mask in consensus_levels.items():
    n_anomalies = sum(mask)
    print(f"\n{level_name} ({n_anomalies} anomalias):")
    if n_anomalies > 0:
        print(data[mask].describe())

# %%
