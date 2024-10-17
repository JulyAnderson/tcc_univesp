import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist



def prepare_data(df, robust=False):
    """
    Prepares data for analysis, including cleaning and normalization.
    :param robust: If True, uses RobustScaler for normalization.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'bool']).columns #aumenta a quantidade de anomalias
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    df_numeric = df[numeric_cols].copy()
    
    # Handle missing values
    df_numeric = df_numeric.fillna(df_numeric.median())
    # Normalize the data
    if robust:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=df_numeric.columns,
        index=df_numeric.index
    )
    
    return df_scaled, df_numeric


def apply_isolation_forest(df_scaled, contamination=0.1):
    """Applies Isolation Forest."""
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    return iso_forest.fit_predict(df_scaled)


def apply_dbscan(df_scaled, eps=0.5, min_samples=5):
    """Applies DBSCAN."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    return dbscan.fit_predict(df_scaled)


def apply_lof(df_scaled, contamination=0.1):
    """Applies Local Outlier Factor."""
    n_neighbors = min(20, len(df_scaled) - 1)
    lof = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors, n_jobs=-1)
    return lof.fit_predict(df_scaled)


def apply_kmeans(df_scaled, n_clusters=2):
    """Applies K-means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(df_scaled)


def plot_results(df_scaled, predictions_dict):
    """Visualizes results."""
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    n_methods = len(predictions_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

    for idx, (method_name, predictions) in enumerate(predictions_dict.items()):
        scatter = axes[idx].scatter(
            df_pca[:, 0],
            df_pca[:, 1],
            c=predictions,
            cmap='RdYlBu',
            alpha=0.6
        )
        axes[idx].set_title(f'{method_name}\n{np.sum(predictions == -1)} anomalies detected')
        axes[idx].set_xlabel('PC1')
        axes[idx].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[idx])

    plt.tight_layout()
    plt.show()


def compare_results(df, predictions_dict):
    """Compares results with additional metrics."""
    results = {}
    for method_name, predictions in predictions_dict.items():
        anomalies = df[predictions == -1]
        results[method_name] = {
            'num_anomalies': len(anomalies),
            'percent_anomalies': (len(anomalies) / len(df)) * 100,
            # Add other metrics here if necessary
        }
    
    return pd.DataFrame(results).T


def dunn_index(predictions, df_scaled):
    """Calculates the Dunn Index."""
    clusters = np.unique(predictions)
    if len(clusters) < 2:
        return 0  # Not enough clusters to compute Dunn Index

    intra_cluster_distances = []
    inter_cluster_distances = []

    for cluster in clusters:
        cluster_points = df_scaled[predictions == cluster]
        if len(cluster_points) > 1:
            # Calculate average intra-cluster distance
            intra_distance = np.mean(cdist(cluster_points, cluster_points, 'euclidean'))
            intra_cluster_distances.append(intra_distance)

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            cluster_i_points = df_scaled[predictions == clusters[i]]
            cluster_j_points = df_scaled[predictions == clusters[j]]
            # Calculate minimum inter-cluster distance
            inter_distance = np.min(cdist(cluster_i_points, cluster_j_points, 'euclidean'))
            inter_cluster_distances.append(inter_distance)

    return min(inter_cluster_distances) / max(intra_cluster_distances) if intra_cluster_distances else 0

def compare_results(df, predictions_dict, df_scaled):
    """Compares results with additional metrics."""
    results = {}
    for method_name, predictions in predictions_dict.items():
        anomalies = df[predictions == -1]
        # Calculate the number of anomalies and percentage
        num_anomalies = len(anomalies)
        percent_anomalies = (num_anomalies / len(df)) * 100
        
        # Calculate additional metrics
        if len(set(predictions)) > 1:  # More than one cluster
            silhouette = silhouette_score(df_scaled, predictions) if len(set(predictions)) > 1 else None
            dunn = dunn_index(predictions, df_scaled)
        else:
            silhouette = None
            dunn = None
            
        results[method_name] = {
            'num_anomalies': num_anomalies,
            'percent_anomalies': percent_anomalies,
            'silhouette_score': silhouette,
            'dunn_index': dunn
        }
    
    return pd.DataFrame(results).T

def main(df):
    """Main function to execute anomaly detection."""
    results_dict = {
        'Isolation Forest': None,
        'DBSCAN': None,
        'LOF': None,
        'K-means': None,
        'DBSCAN Tuning': None,
        'K-means Tuning': None
    }

    print("Preparing data with Standard Scaler...")
    df_scaled_standard, df_numeric = prepare_data(df, robust=False)

    print("Applying anomaly detection methods with Standard Scaler...")
    results_dict['Isolation Forest'] = apply_isolation_forest(df_scaled_standard)
    results_dict['DBSCAN'] = apply_dbscan(df_scaled_standard)
    results_dict['LOF'] = apply_lof(df_scaled_standard)
    results_dict['K-means'] = apply_kmeans(df_scaled_standard)

    print("Preparing data with Robust Scaler...")
    df_scaled_robust, _ = prepare_data(df, robust=True)

    print("Applying tuning and anomaly detection methods with Robust Scaler...")
    results_dict['DBSCAN Tuning'] = apply_dbscan(df_scaled_robust, eps=0.5, min_samples=5)
    results_dict['K-means Tuning'] = apply_kmeans(df_scaled_robust, n_clusters=2)

    # Clean None results
    results_dict = {k: v for k, v in results_dict.items() if v is not None}

    print("Plotting results...")
    plot_results(df_scaled_standard, results_dict)

    print("Comparing results...")
    comparison = compare_results(df_numeric, results_dict, df_scaled_standard)
    print("\nComparison of methods:")
    print(comparison)

    return results_dict, comparison



if __name__ == "__main__":
    try:
        df_reduzido = pd.read_csv('dados_processados.csv')
        predictions, comparison = main(df_reduzido)

    except Exception as e:
        print(f"Error loading or processing data: {e}")

### Analise dos resultados Principais Métricas Explicadas
'''
Comparison of methods:
                  num_anomalies  percent_anomalies  silhouette_score  \
Isolation Forest          219.0          10.009141          0.277084   
DBSCAN                     11.0           0.502742          0.691962   
LOF                       219.0          10.009141          0.257367   
K-means                     0.0           0.000000          0.757184   
DBSCAN Tuning              65.0           2.970750          0.745327   
K-means Tuning              0.0           0.000000          0.953357   

                  dunn_index  
Isolation Forest    0.000735  
DBSCAN              0.039305  
LOF                 0.000011  
K-means             2.178960  
DBSCAN Tuning       0.002791  
K-means Tuning      2.327383  


Interpretação: Detecção Moderada de Anomalias:

DBSCAN Tuning detectou 65 anomalias, representando aproximadamente 3% dos dados. Isso indica que ele é capaz de identificar anomalias sem ser excessivamente sensível, ao contrário de métodos como Isolation Forest e LOF, que detectaram uma quantidade considerável de anomalias (10%).
Qualidade dos Clusters:

Apresentou um silhouette_score de 0.745327, o que indica uma boa definição dos grupos. Este é um valor alto, sugerindo que o modelo foi capaz de criar clusters bem definidos, o que é importante para separar efetivamente os dados normais dos anômalos.
Dunn Index:

Embora o dunn_index de 0.002791 não seja o mais alto entre os métodos avaliados, ele é aceitável para o DBSCAN, que lida bem com clusters de forma arbitrária. Isso implica que os clusters são razoavelmente separados e evita a criação de anomalias falsas por conta de clusters próximos demais.
Flexibilidade de Parâmetros:

A versão ajustada (tuning) do DBSCAN oferece maior precisão na identificação de outliers em comparação com a versão padrão, melhorando tanto a quantidade quanto a qualidade da detecção.

'''
