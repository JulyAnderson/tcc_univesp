#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score,silhouette_samples
import seaborn as sns


class AnomalyDetectionEnsemble:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.dbscan = None
        self.reconstruction_threshold = None
        
    def create_autoencoder(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        
        encoded = Dense(int(input_dim * 0.75), activation='relu')(input_layer)
        encoded = Dense(int(input_dim * 0.5), activation='relu')(encoded)
        encoded = Dense(int(input_dim * 0.33), activation='relu')(encoded)
        
        decoded = Dense(int(input_dim * 0.5), activation='relu')(encoded)
        decoded = Dense(int(input_dim * 0.75), activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
    def fit(self, X):
        # Pré-processamento
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
        
        # Treinando o Autoencoder
        print("Treinando Autoencoder...")
        self.create_autoencoder(X.shape[1])
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculando erro de reconstrução
        reconstructed = self.autoencoder.predict(X_scaled)
        reconstruction_errors = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        self.reconstruction_threshold = np.percentile(reconstruction_errors, 
                                                    (1 - self.contamination) * 100)
        
        # Otimizando parâmetros do DBSCAN
        print("\nOtimizando DBSCAN...")
        best_score = -np.inf
        best_eps = 0.5  # Valor padrão inicial
        
        eps_range = np.linspace(0.1, 1.0, 10)
        for eps in eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_scaled)
            
            # Verificar se há pelo menos dois clusters (excluindo ruído)
            valid_labels = labels[labels != -1]
            if len(np.unique(valid_labels)) >= 2:
                try:
                    score = silhouette_score(X_scaled[labels != -1], valid_labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                except:
                    continue
        
        print(f"Melhor eps encontrado: {best_eps:.3f}")
        
        # Treinando DBSCAN com o melhor eps encontrado
        self.dbscan = DBSCAN(eps=best_eps, min_samples=5)
        self.dbscan.fit(X_scaled)
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        
        # Predições do Autoencoder
        reconstructed = self.autoencoder.predict(X_scaled)
        reconstruction_errors = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        ae_predictions = (reconstruction_errors > self.reconstruction_threshold).astype(int)
        
        # Predições do DBSCAN
        dbscan_predictions = self.dbscan.fit_predict(X_scaled)
        dbscan_predictions = (dbscan_predictions == -1).astype(int)
        
        # Combinando as predições (OR lógico)
        ensemble_predictions = np.logical_or(ae_predictions, dbscan_predictions).astype(int)
        
        return ensemble_predictions, reconstruction_errors, dbscan_predictions
    
    def plot_results(self, X, predictions, reconstruction_errors, dbscan_predictions):
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Visualization of anomalies
        plt.subplot(131)
        plt.scatter(range(len(X)), reconstruction_errors, c=predictions, cmap='viridis')
        plt.axhline(y=self.reconstruction_threshold, color='r', linestyle='--')
        plt.title('Anomalias Detectadas')
        plt.xlabel('Índice da Amostra')
        plt.ylabel('Erro de Reconstrução')
        
        # Plot 2: Comparison of methods
        plt.subplot(132)
        comparison_df = pd.DataFrame({
            'Autoencoder': (reconstruction_errors > self.reconstruction_threshold).astype(int),
            'DBSCAN': dbscan_predictions,
            'Ensemble': predictions
        })
        sns.heatmap(comparison_df.T, cmap='YlOrRd', cbar=False)
        plt.title('Comparação dos Métodos')
        
        # Plot 3: Distribution of reconstruction errors
        plt.subplot(133)
        sns.histplot(reconstruction_errors, bins=50)
        plt.axvline(x=self.reconstruction_threshold, color='r', linestyle='--')
        plt.title('Distribuição dos Erros de Reconstrução')
        
        plt.tight_layout()
        plt.show()

#%%

def main():
    # 1. Carregando os dados
    print("Carregando dados...")
    X = load_data('dados_processados.csv')
    print(f"Shape dos dados: {X.shape}")
    print(f"Colunas disponíveis: {X.columns.tolist()}")
    
    # 2. Criando e treinando o detector
    print("\nInicializando detector de anomalias...")
    detector = AnomalyDetectionEnsemble(contamination=0.1)  # 10% dos dados serão considerados anomalias
    
    # 3. Treinando o modelo
    print("\nIniciando treinamento...")
    detector.fit(X)
    
    # 4. Fazendo predições
    print("\nRealizando predições...")
    predictions, reconstruction_errors, dbscan_predictions = detector.predict(X)
    
    # 5. Adicionando resultados ao DataFrame original
    results_df = X.copy()
    results_df['anomalia'] = predictions
    results_df['erro_reconstrucao'] = reconstruction_errors
    results_df['anomalia_dbscan'] = dbscan_predictions
    
    # 6. Salvando resultados
    print("\nSalvando resultados...")
    results_df.to_csv('resultados_anomalias.csv', index=False)
    
    # 7. Exibindo algumas estatísticas
    print("\nEstatísticas das anomalias detectadas:")
    print(f"Total de amostras: {len(predictions)}")
    print(f"Número de anomalias detectadas: {sum(predictions)}")
    print(f"Percentual de anomalias: {(sum(predictions)/len(predictions))*100:.2f}%")
    
    # 8. Plotando os resultados
    print("\nGerando visualizações...")
    detector.plot_results(X, predictions, reconstruction_errors, dbscan_predictions)

if __name__ == "__main__":
    main()

# %%
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_silhouette(X, labels):
    """
    Calcula e visualiza o Silhouette Score para um conjunto de dados clusterizado
    
    Parâmetros:
    X : array-like de shape (n_samples, n_features)
        Dados de entrada
    labels : array-like de shape (n_samples,)
        Rótulos dos clusters atribuídos pelo algoritmo
    """
    # Remove pontos classificados como ruído (-1)
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]
    
    # Calcula o score geral
    if len(np.unique(labels_valid)) > 1:
        sil_score = silhouette_score(X_valid, labels_valid)
        
        # Calcula os scores individuais
        sample_silhouette_values = silhouette_samples(X_valid, labels_valid)
        
        # Visualização
        plt.figure(figsize=(10, 6))
        
        # Plot do silhouette score para cada amostra
        plt.plot(sample_silhouette_values)
        plt.axhline(y=sil_score, color="red", linestyle="--", 
                   label=f'Média: {sil_score:.3f}')
        
        plt.title('Silhouette Score por Amostra')
        plt.xlabel('Amostras')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.show()
        
        return sil_score
    else:
        print("Erro: É necessário ter pelo menos 2 clusters para calcular o Silhouette Score")
        return None
    
# Exemplo de uso
X_scaled = scaler.transform(X)  # seus dados escalados
labels = detector.dbscan.labels_  # labels do DBSCAN

sil_score = calculate_silhouette(X_scaled, labels)
print(f"Silhouette Score Global: {sil_score}")
