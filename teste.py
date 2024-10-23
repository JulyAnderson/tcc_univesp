# Vamos carregar o arquivo dados_processados.pkl para verificar seu conteúdo.
import pickle

# Carregar o arquivo
file_path = 'dados/processados/dados_processados.pkl'

with open(file_path, 'rb') as file:
    dados_processados = pickle.load(file)

# Verificando os primeiros dados para entender a estrutura
dados_processados.head()


from sklearn.preprocessing import StandardScaler

# Selecionar as colunas numéricas para padronização
numerical_columns = ['valorTotalHomologado', 'anoCompra', 'sequencialCompra', 
                     'valorTotalEstimado', 'unidadeOrgao.codigoIbge']

# Inicializar o scaler
scaler = StandardScaler()

# Padronizar os dados numéricos
dados_processados_scaled = dados_processados.copy()
dados_processados_scaled[numerical_columns] = scaler.fit_transform(dados_processados[numerical_columns])

# Verificando o resultado da padronização
dados_processados_scaled[numerical_columns].head()


from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
import numpy as np

# Criar uma função para aplicar os modelos de anomalias
def detect_anomalies_with_models(data):
    results = {}

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(data[numerical_columns])
    results['DBSCAN'] = dbscan_labels

    # Isolation Forest
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    if_labels = isolation_forest.fit_predict(data[numerical_columns])
    results['IsolationForest'] = if_labels

    # LOF (Local Outlier Factor)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof_labels = lof.fit_predict(data[numerical_columns])
    results['LOF'] = lof_labels

    # K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(data[numerical_columns])
    distances = kmeans.transform(data[numerical_columns])
    kmeans_labels = np.where(np.min(distances, axis=1) > np.percentile(np.min(distances, axis=1), 90), -1, 1)
    results['KMeans'] = kmeans_labels

    return results

# Aplicar os modelos e detectar anomalias
model_results = detect_anomalies_with_models(dados_processados_scaled)

# Verificando os resultados para os primeiros registros
model_results


import matplotlib.pyplot as plt

# Contagem de anomalias detectadas por cada modelo
anomalias_detectadas = {
    'DBSCAN': np.sum(model_results['DBSCAN'] == -1),
    'IsolationForest': np.sum(model_results['IsolationForest'] == -1),
    'LOF': np.sum(model_results['LOF'] == -1),
    'KMeans': np.sum(model_results['KMeans'] == -1)
}

# Plotar o número de anomalias detectadas por cada modelo
plt.figure(figsize=(10, 6))
plt.bar(anomalias_detectadas.keys(), anomalias_detectadas.values(), color=['blue', 'green', 'red', 'purple'])
plt.title('Número de Anomalias Detectadas por Modelo')
plt.xlabel('Modelo')
plt.ylabel('Número de Anomalias')
plt.show()

# Exibir os resultados
anomalias_detectadas


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler


X = dados_processados_scaled[numerical_columns].values


input_dim = X.shape[1]  # Número de colunas no conjunto de dados
encoding_dim = 5  # Número de neurônios na camada escondida

# Camada de entrada
input_layer = Input(shape=(input_dim,))

# Encoder
encoder = Dense(encoding_dim, activation="relu")(input_layer)

# Decoder
decoder = Dense(input_dim, activation="linear")(encoder)

# Definindo o Autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento do modelo
autoencoder.fit(X, X, 
                epochs=50, 
                batch_size=32, 
                shuffle=True, 
                validation_split=0.1)

# Prever as reconstruções do Autoencoder
X_pred = autoencoder.predict(X)

# Calcular o erro de reconstrução (MSE)
reconstruction_error = np.mean(np.power(X - X_pred, 2), axis=1)

threshold = np.percentile(reconstruction_error, 90)

# Detectar anomalias
anomalies = reconstruction_error > threshold
print(f"Anomalias detectadas: {np.sum(anomalies)}")
