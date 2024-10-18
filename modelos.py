import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ModelConfig:
    """Configurações para os modelos de detecção de anomalias."""
    isolation_forest_contamination: float = 0.1
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    lof_contamination: float = 0.1
    kmeans_n_clusters: int = 2
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Classe para preprocessamento de dados com suporte a colunas temporais."""
    
    def __init__(self, robust: bool = False):
        self.robust = robust
        self.scaler = RobustScaler() if robust else StandardScaler()
        self.datetime_columns = []
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        X_processed = self._preprocess_data(X)
        self.feature_names_ = X_processed.columns.tolist()
        self.scaler.fit(X_processed)
        return self
        
    def transform(self, X):
        X_processed = self._preprocess_data(X)
        
        if self.feature_names_ is not None:
            missing_cols = set(self.feature_names_) - set(X_processed.columns)
            for col in missing_cols:
                X_processed[col] = 0
            X_processed = X_processed[self.feature_names_]
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        return X_scaled
    
    def _preprocess_data(self, X):
        X_processed = X.copy()
        
        # Identificar colunas datetime
        self.datetime_columns = [
            col for col in X_processed.columns 
            if pd.api.types.is_datetime64_any_dtype(X_processed[col]) or
               (isinstance(X_processed[col].iloc[0], str) and self._is_datetime(X_processed[col].iloc[0]))
        ]
        
        # Converter colunas datetime
        for col in self.datetime_columns:
            try:
                X_processed[col] = pd.to_datetime(X_processed[col])
                timestamp_col = f'{col}_timestamp'
                X_processed[timestamp_col] = X_processed[col].astype(np.int64) // 10**9
                X_processed = X_processed.drop(columns=[col])
            except Exception as e:
                logging.warning(f"Erro ao processar coluna datetime {col}: {e}")
        
        # Processar colunas numéricas
        numeric_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
        X_numeric = X_processed[numeric_cols].copy()
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        return X_numeric
    
    @staticmethod
    def _is_datetime(value: str) -> bool:
        try:
            pd.to_datetime(value)
            return True
        except:
            return False

class AnomalyDetector:
    """Classe principal para detecção de anomalias usando múltiplos métodos."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.preprocessor = None
        self.best_model = None
        self.best_model_name = None
        self.metrics_history = {}
        self.original_columns = None
    
    def calculate_dunn_index(self, labels: np.ndarray, X: pd.DataFrame) -> float:
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values
                
            distances = pdist(X)
            dist_matrix = squareform(distances)
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            min_inter_cluster = float('inf')
            max_intra_cluster = float('-inf')
            
            for i in range(len(unique_labels)):
                cluster_i = X[labels == unique_labels[i]]
                if len(cluster_i) > 1:
                    intra_dist = np.max(pdist(cluster_i))
                    max_intra_cluster = max(max_intra_cluster, intra_dist)
                
                for j in range(i + 1, len(unique_labels)):
                    cluster_j = X[labels == unique_labels[j]]
                    if len(cluster_i) > 0 and len(cluster_j) > 0:
                        inter_dist = cdist(cluster_i, cluster_j).min()
                        min_inter_cluster = min(min_inter_cluster, inter_dist)
            
            if max_intra_cluster == float('-inf') or min_inter_cluster == float('inf'):
                return 0.0
                
            return min_inter_cluster / max_intra_cluster if max_intra_cluster > 0 else 0.0
            
        except Exception as e:
            logging.warning(f"Erro ao calcular índice Dunn: {e}")
            return 0.0
    
    def create_pipeline(self, robust: bool = False) -> Dict[str, Pipeline]:
        preprocessor = DataPreprocessor(robust=robust)
        
        pipelines = {
            'IsolationForest': Pipeline([
                ('preprocessor', preprocessor),
                ('model', IsolationForest(
                    contamination=self.config.isolation_forest_contamination,
                    random_state=self.config.random_state,
                    n_jobs=-1
                ))
            ]),
            'DBSCAN': Pipeline([
                ('preprocessor', preprocessor),
                ('model', DBSCAN(
                    eps=self.config.dbscan_eps,
                    min_samples=self.config.dbscan_min_samples,
                    n_jobs=-1
                ))
            ]),
            'LOF': Pipeline([
                ('preprocessor', preprocessor),
                ('model', LocalOutlierFactor(
                    contamination=self.config.lof_contamination,
                    novelty=True,
                    n_jobs=-1
                ))
            ]),
            'KMeans': Pipeline([
                ('preprocessor', preprocessor),
                ('model', KMeans(
                    n_clusters=self.config.kmeans_n_clusters,
                    random_state=self.config.random_state,
                    n_init=10
                ))
            ])
        }
        return pipelines
    
    def select_best_model(self, results: Dict[str, Dict[str, float]]) -> None:
        best_score = float('-inf')
        best_model_name = None
        
        for name, metrics in results.items():
            combined_score = (
                0.7 * metrics['mean_silhouette'] +
                0.3 * metrics['mean_dunn']
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_model_name = name
        
        if best_model_name is not None:
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            logging.info(f"Melhor modelo selecionado: {best_model_name}")
        else:
            self.best_model_name = 'IsolationForest'
            self.best_model = self.models['IsolationForest']
            logging.warning("Nenhum modelo teve bom desempenho. Usando IsolationForest como fallback.")
    
    def train_validate_models(self, X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        logging.info("Iniciando treinamento e validação dos modelos...")
        self.original_columns = X.columns.tolist()
        
        pipelines = self.create_pipeline()
        results = {}
        
        X_train, X_test = train_test_split(
            X,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        kf = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)
        
        for name, pipeline in pipelines.items():
            logging.info(f"Treinando modelo: {name}")
            
            try:
                pipeline.fit(X_train)
                self.preprocessor = pipeline.named_steps['preprocessor']
                
                fold_metrics = {
                    'silhouette_scores': [],
                    'dunn_indices': []
                }
                
                for train_idx, val_idx in kf.split(X_train):
                    X_fold_train = X_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    
                    pipeline.fit(X_fold_train)
                    
                    X_fold_val_processed = pipeline.named_steps['preprocessor'].transform(X_fold_val)
                    
                    if name == 'DBSCAN':
                        predictions = pipeline.named_steps['model'].fit_predict(X_fold_val_processed)
                    else:
                        predictions = pipeline.predict(X_fold_val)
                    
                    if len(set(predictions)) > 1:
                        fold_metrics['silhouette_scores'].append(
                            silhouette_score(X_fold_val_processed, predictions)
                        )
                        fold_metrics['dunn_indices'].append(
                            self.calculate_dunn_index(predictions, X_fold_val_processed)
                        )
                
                results[name] = {
                    'mean_silhouette': np.mean(fold_metrics['silhouette_scores']) if fold_metrics['silhouette_scores'] else float('-inf'),
                    'std_silhouette': np.std(fold_metrics['silhouette_scores']) if fold_metrics['silhouette_scores'] else 0,
                    'mean_dunn': np.mean(fold_metrics['dunn_indices']) if fold_metrics['dunn_indices'] else float('-inf'),
                    'std_dunn': np.std(fold_metrics['dunn_indices']) if fold_metrics['dunn_indices'] else 0
                }
                
                self.models[name] = pipeline
                
            except Exception as e:
                logging.error(f"Erro ao treinar {name}: {e}")
                results[name] = {
                    'mean_silhouette': float('-inf'),
                    'std_silhouette': 0,
                    'mean_dunn': float('-inf'),
                    'std_dunn': 0
                }
        
        self.select_best_model(results)
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Nenhum modelo treinado. Execute train_validate_models primeiro.")
            
        if self.original_columns:
            X = X[self.original_columns]
            
        if self.best_model_name == 'DBSCAN':
            return self.best_model.named_steps['model'].fit_predict(
                self.best_model.named_steps['preprocessor'].transform(X)
            )
        else:
            return self.best_model.predict(X)

    def save_model(self, path: str = "models") -> None:
        if self.best_model is None:
            raise ValueError("Nenhum modelo para salvar.")
        
        Path(path).mkdir(exist_ok=True)
        
        with open(f"{path}/best_model.pkl", 'wb') as model_file:
            pickle.dump(self.best_model, model_file)
        
        with open(f"{path}/config.pkl", 'wb') as config_file:
            pickle.dump(self.config, config_file)
        
        logging.info(f"Modelo e configuração salvos em {path}")

    def visualize_results(self, X: pd.DataFrame, predictions: np.ndarray) -> None:
        pca = PCA(n_components=2)
        X_processed = self.best_model.named_steps['preprocessor'].transform(X)
        X_pca = pca.fit_transform(X_processed)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=predictions,
            cmap='RdYlBu',
            alpha=0.6
        )
        plt.title(f'Resultados da Detecção de Anomalias\nMétodo: {self.best_model_name}')
        plt.xlabel('Primeira Componente Principal')
        plt.ylabel('Segunda Componente Principal')
        plt.colorbar(scatter, label='Classificação')
        plt.grid(True, alpha=0.3)
        
        anomaly_count = np.sum(predictions == -1) if -1 in predictions else 0
        total_count = len(predictions)
        plt.text(
            0.02, 0.98,
            f'Total de pontos: {total_count}\nAnomalias detectadas: {anomaly_count} ({anomaly_count/total_count:.1%})',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        viz_path = f'anomaly_detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Visualização salva em {viz_path}")

    @staticmethod
    def calculate_statistics(df: pd.DataFrame, predictions: np.ndarray) -> None:
        df_copy = df.copy()
        df_copy['anomalia'] = predictions
        
        normal_data = df_copy[df_copy['anomalia'] != -1]
        anomaly_data = df_copy[df_copy['anomalia'] == -1]

        # Calcular estatísticas apenas para colunas numéricas
        numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
        normal_stats = normal_data[numeric_cols].describe()
        anomaly_stats = anomaly_data[numeric_cols].describe()

        print("\nEstatísticas dos Dados Normais:")
        print(normal_stats)
        print("\nEstatísticas das Anomalias:")
        print(anomaly_stats)

        # Salvar estatísticas em arquivo
        stats_path = f'statistics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(stats_path, 'w') as f:
            f.write("Estatísticas dos Dados Normais:\n")
            f.write(normal_stats.to_string())
            f.write("\n\nEstatísticas das Anomalias:\n")
            f.write(anomaly_stats.to_string())

        logging.info(f"Estatísticas salvas em {stats_path}")

    @staticmethod
    def visualize_distributions(df: pd.DataFrame, predictions: np.ndarray) -> None:
        """Visualiza a distribuição das características para dados normais e anômalos."""
        df_copy = df.copy()
        df_copy['anomalia'] = predictions
        
        numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'anomalia']
        
        for col in numeric_cols:
            plt.figure(figsize=(12, 6))
            
            # Plotar distribuição normal
            sns.kdeplot(
                data=df_copy[df_copy['anomalia'] != -1][col],
                label='Normal',
                color='blue'
            )
            
            # Plotar distribuição das anomalias
            if len(df_copy[df_copy['anomalia'] == -1]) > 0:
                sns.kdeplot(
                    data=df_copy[df_copy['anomalia'] == -1][col],
                    label='Anomalia',
                    color='red'
                )
            
            plt.title(f'Distribuição de {col}')
            plt.xlabel(col)
            plt.ylabel('Densidade')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Salvar visualização
            dist_path = f'distribution_{col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Distribuição de {col} salva em {dist_path}")

    @staticmethod
    def visualize_boxplots(df: pd.DataFrame, predictions: np.ndarray) -> None:
        """Cria boxplots comparativos entre dados normais e anômalos."""
        df_copy = df.copy()
        df_copy['anomalia'] = predictions
        
        numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'anomalia']
        
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            
            # Criar dados para o boxplot
            data_to_plot = [
                df_copy[df_copy['anomalia'] != -1][col],
                df_copy[df_copy['anomalia'] == -1][col]
            ]
            
            # Criar boxplot
            plt.boxplot(data_to_plot, labels=['Normal', 'Anomalia'])
            plt.title(f'Boxplot de {col}')
            plt.ylabel(col)
            plt.grid(True, alpha=0.3)
            
            # Salvar visualização
            box_path = f'boxplot_{col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(box_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Boxplot de {col} salvo em {box_path}")

def main(input_file: str) -> None:
    """Função principal para executar o pipeline completo."""
    try:
        # Carregar dados
        df = pd.read_csv(input_file)
        logging.info(f"Dados carregados com sucesso: {df.shape}")
        
        # Configurar detector
        config = ModelConfig()
        detector = AnomalyDetector(config)
        
        # Treinar e validar modelos
        results = detector.train_validate_models(df)
        
        # Realizar predições com o melhor modelo
        predictions = detector.predict(df)
        
        # Visualizar resultados
        detector.visualize_results(df, predictions)
        
        # Calcular e visualizar estatísticas
        detector.calculate_statistics(df, predictions)
        
        # Visualizar distribuições
        detector.visualize_distributions(df, predictions)
        
        # Visualizar boxplots
        detector.visualize_boxplots(df, predictions)
        
        # Salvar modelo
        detector.save_model()
        
        # Salvar resultados da comparação dos modelos
        results_df = pd.DataFrame(results).T
        results_path = f'model_comparison_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(results_path)
        
        # Salvar predições
        predictions_df = pd.DataFrame({
            'index': df.index,
            'anomalia': predictions
        })
        predictions_path = f'anomalias_detectadas_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        predictions_df.to_csv(predictions_path, index=False)
        
        logging.info(f"""
        Análise concluída com sucesso!
        Melhor modelo: {detector.best_model_name}
        Resultados dos modelos salvos em: {results_path}
        Anomalias detectadas salvas em: {predictions_path}
        """)
        
    except Exception as e:
        logging.error(f"Erro durante a análise: {e}")
        raise

if __name__ == "__main__":
    main('dados_processados.csv')