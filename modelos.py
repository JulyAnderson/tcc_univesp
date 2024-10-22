import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path
import pickle
from datetime import datetime
import os

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
    def __init__(self):
        # Parâmetros gerais
        self.random_state = 42
        
        # Parâmetros para IsolationForest
        self.isolation_forest_contamination = 0.1
        
        # Parâmetros para DBSCAN
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 5
        
        # Parâmetros para LOF
        self.lof_n_neighbors = 20
        self.lof_contamination = 0.1
        
        # Parâmetros para KMeans
        self.kmeans_n_clusters = 2

class ModelPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = None
        self.datetime_columns_ = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.feature_names_ = X.columns.tolist()        
        numeric_columns = self.features_names_.select_dtypes(include=['int64', 'float64', "bool"]).columns
        self.scaler.fit(self.feature_names_[numeric_columns])
        return self
        
    def transform(self, X):
        X_transformed = self._convert_datetime(X.copy())
        numeric_columns = X_transformed.select_dtypes(include=['int64', 'float64']).columns
        X_transformed[numeric_columns] = self.scaler.transform(X_transformed[numeric_columns])
        return X_transformed

    
class AnomalyDetector:
    """Classe principal para detecção de anomalias usando múltiplos métodos."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.preprocessor = ModelPreprocessor()
        self.best_model = None
        self.best_model_name = None
        self.metrics_history = {}
        self.original_columns = None

    def create_pipeline(self):
        pipelines = {
            'IsolationForest': Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', IsolationForest(
                    contamination=self.config.isolation_forest_contamination,
                    random_state=self.config.random_state,
                    n_jobs=-1
                ))
            ]),
            'DBSCAN': Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', DBSCAN(
                    eps=self.config.dbscan_eps,
                    min_samples=self.config.dbscan_min_samples,
                    n_jobs=-1
                ))
            ]),
            'LOF': Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', LocalOutlierFactor(
                    n_neighbors=self.config.lof_n_neighbors,
                    contamination=self.config.lof_contamination,
                    n_jobs=-1
                ))
            ]),
            'KMeans': Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', KMeans(
                    n_clusters=self.config.kmeans_n_clusters,
                    random_state=self.config.random_state,
                    n_init=10
                ))
            ])
        }
        return pipelines

    def calculate_dunn_index(self, labels: np.ndarray, X: pd.DataFrame) -> float:
        """Calcula o índice Dunn para avaliar a qualidade do clustering."""
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

    def train_validate_models(self, X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        logging.info(f"Iniciando treinamento e validação dos modelos...")
        logging.info(f"Shape dos dados de entrada: {X.shape}")
        logging.info(f"Tipos de dados:\n{X.dtypes}")
        logging.info(f"Resumo estatístico:\n{X.describe()}")
        logging.info(f"Colunas: {X.columns.tolist()}")
        
        self.original_columns = X.columns.tolist()
        pipelines = self.create_pipeline()
        results = {}
        self.models.clear()  # Limpa modelos anteriores

        for name, pipeline in pipelines.items():
            logging.info(f"Tentando treinar modelo: {name}")
            try:
                # Pré-processamento dos dados
                logging.info(f"Iniciando pré-processamento para {name}")
                X_preprocessed = pipeline.named_steps['preprocessor'].fit_transform(X)
                logging.info(f"Pré-processamento concluído para {name}. Shape: {X_preprocessed.shape}")
                
                # Treinamento do modelo
                model = pipeline.named_steps['model']
                if hasattr(model, 'fit_predict'):
                    logging.info(f"Usando fit_predict para {name}")
                    y_pred = model.fit_predict(X_preprocessed)
                else:
                    logging.info(f"Usando fit e predict separadamente para {name}")
                    model.fit(X_preprocessed)
                    y_pred = model.predict(X_preprocessed)
                
                logging.info(f"Modelo {name} treinado com sucesso")
                self.models[name] = pipeline

                # Calcular métricas
                metrics = self.calculate_metrics(y_pred)
                results[name] = metrics
                logging.info(f"Métricas calculadas para {name}: {metrics}")
            except Exception as e:
                logging.error(f"Erro ao treinar {name}: {str(e)}")
                logging.error("Traceback completo:", exc_info=True)

        if not self.models:
            logging.error("Nenhum modelo foi treinado com sucesso.")
            raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

        return results

    def select_best_model(self, results: Dict[str, Dict[str, float]]) -> None:
        """Seleciona o melhor modelo com base nas métricas de avaliação."""
        best_score = float('-inf')
        best_model_name = None
        
        # Verifica se há modelos treinados disponíveis
        if not self.models:
            logging.warning("Nenhum modelo foi treinado com sucesso. Retreinando IsolationForest...")
            try:
                # Cria o pipeline sem passar o preprocessor como argumento
                pipeline = self.create_pipeline()['IsolationForest']
                X_dummy = pd.DataFrame(np.random.randn(10, 2))  # Dados dummy para treino
                pipeline.fit(X_dummy)
                self.models['IsolationForest'] = pipeline
                self.best_model_name = 'IsolationForest'
                self.best_model = pipeline
                return
            except Exception as e:
                logging.error(f"Erro ao treinar modelo fallback: {e}")
                raise RuntimeError("Não foi possível treinar nenhum modelo.")
        
        for name, metrics in results.items():
            if name in self.models:  # Só considera modelos que foram treinados com sucesso
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
            # Se nenhum modelo teve bom desempenho, mas temos modelos treinados
            fallback_model = next(iter(self.models.keys()))
            self.best_model_name = fallback_model
            self.best_model = self.models[fallback_model]
            logging.warning(f"Nenhum modelo teve bom desempenho. Usando {fallback_model} como fallback.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predições usando o melhor modelo selecionado."""
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
        """Salva o melhor modelo e suas configurações."""
        if self.best_model is None:
            raise ValueError("Nenhum modelo para salvar.")
        
        Path(path).mkdir(exist_ok=True)
        
        with open(f"{path}/best_model.pkl", 'wb') as model_file:
            pickle.dump(self.best_model, model_file)
        
        with open(f"{path}/config.pkl", 'wb') as config_file:
            pickle.dump(self.config, config_file)
        
        logging.info(f"Modelo e configuração salvos em {path}")

    def visualize_results(self, X: pd.DataFrame, predictions: np.ndarray) -> None:
        """Visualiza os resultados da detecção de anomalias usando PCA."""
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

def main(input_file: str) -> None:
    """
    Função principal para executar o pipeline completo de detecção de anomalias.
    
    Args:
        input_file (str): Caminho para o arquivo de entrada
    """
    try:
    # Definir arquivo de saída
        output_dir = '/dados/processados'

        # Verificar se o diretório existe, caso contrário, criá-lo
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Carregar o DataFrame do arquivo pickle
        logging.info("Iniciando a carga dos dados do arquivo pickle...")
        with open(input_file, 'rb') as file:
            processed_df = pickle.load(file)
        
        logging.info(f"Dados carregados com sucesso. Shape final: {processed_df.shape}")
        
        # Remover colunas de data antes da detecção de anomalias
        date_columns = [
            'dataAberturaProposta', 'dataEncerramentoProposta',
            'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao'
        ]
        processed_df.drop(columns=date_columns, errors='ignore', inplace=True)
        logging.info(f"Colunas de data removidas. Shape final: {processed_df.shape}")


        # Configurar e executar detector de anomalias
        logging.info("Iniciando detecção de anomalias...")
        config = ModelConfig()
        detector = AnomalyDetector(config)
        
        # Treinar e validar modelos
        results = detector.train_validate_models(processed_df)
        
        if detector.best_model is None:
            raise RuntimeError("Nenhum modelo foi treinado com sucesso.")
        
        # Realizar predições
        predictions = detector.predict(processed_df)
        
        # Visualizar resultados
        detector.visualize_results(processed_df, predictions)
        
        # Salvar modelo
        detector.save_model(path='models')
        
        logging.info(f"""
        Análise concluída com sucesso!
        Melhor modelo: {detector.best_model_name}
        """)
        
    except Exception as e:
        logging.error(f"Erro durante a análise: {e}")
        raise
if __name__ == "__main__":
    input_file = 'dados/processados/dados_processados.pkl'  # Altere para o caminho do seu arquivo pickle
    main(input_file)