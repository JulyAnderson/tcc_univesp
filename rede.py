import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns

class AnomalyDetectionEnsemble:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.dbscan = None
        self.reconstruction_threshold = None
        
    def create_autoencoder(self, input_dim):
        # Definindo a arquitetura do autoencoder
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(int(input_dim * 0.75), activation='relu')(input_layer)
        encoded = Dense(int(input_dim * 0.5), activation='relu')(encoded)
        encoded = Dense(int(input_dim * 0.33), activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(int(input_dim * 0.5), activation='relu')(encoded)
        decoded = Dense(int(input_dim * 0.75), activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Criando o modelo
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
        best_score = -1
        best_eps = None
        
        for eps in np.linspace(0.1, 1.0, 10):
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_scaled)
            
            # Ignorando pontos de ruído para o cálculo do silhouette score
            if len(np.unique(labels)) > 1 and -1 not in labels:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
        
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