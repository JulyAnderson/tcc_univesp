# %%
import pandas as pd 
import numpy as np
import seaborn as sns

dados = pd.read_csv("dados/brutos/dados_colunas_normalizadas.csv", index_col="Unnamed: 0")


dados = dados.drop_duplicates()

#  0   valorTotalHomologado               2543 non-null   float64
#  21  valorTotalEstimado                 2967 non-null   float64
#%%

# Estatísticas descritivas da coluna valorTotalHomologado
dados['valorTotalHomologado'].describe()

import matplotlib.pyplot as plt

# Criar um histograma da coluna valorTotalHomologado
dados['valorTotalHomologado'].hist(bins=10)  # bins define o número de "caixas" no gráfico
plt.title('Distribuição de valorTotalHomologado')
plt.xlabel('Valor Total Homologado')
plt.ylabel('Frequência')
plt.show()

# Criar um boxplot
dados.boxplot(column='valorTotalHomologado')
plt.title('Boxplot de valorTotalHomologado')
plt.show()
#%%
# Calcular os quartis
Q1 = dados['valorTotalHomologado'].quantile(0.25)  # 1º Quartil (25%)
Q3 = dados['valorTotalHomologado'].quantile(0.75)  # 3º Quartil (75%)
IQR = Q3 - Q1  # Intervalo Interquartil (IQR)

# Definir limites para outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Exibir os limites
print(f'Limite Inferior: {limite_inferior}')
print(f'Limite Superior: {limite_superior}')

# Filtrar os outliers
outliers = dados[(dados['valorTotalHomologado'] < limite_inferior) | (dados['valorTotalHomologado'] > limite_superior)]

# Exibir os outliers
outliers.shape #259 outiers
outliers["valorTotalHomologado"].describe()

# 
dados["diff_homolagado_estimado"] = dados["valorTotalHomologado"]-dados["valorTotalEstimado"]
dados["diff_homolagado_estimado"].isnull().sum()

correlacao = dados[["valorTotalHomologado","valorTotalEstimado","diff_homolagado_estimado"]].corr() #0,970389

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configurações gerais do plot
plt.style.use('default')  # Usando estilo default do matplotlib
sns.set_theme(style="whitegrid")  # Aplicando tema do seaborn
sns.set_palette("husl")

# Criar figura com tamanho personalizado
plt.figure(figsize=(12, 6))

# Criar boxplot customizado
ax = sns.boxplot(y=dados['valorTotalHomologado'], 
                width=0.5,
                color='lightblue',
                showfliers=True,
                fliersize=5,
                flierprops=dict(marker='o', 
                              markerfacecolor='red',
                              markersize=4,
                              alpha=0.5))

# Adicionar um violinplot para mostrar a distribuição dos dados
sns.violinplot(y=dados['valorTotalHomologado'],
               color='lightgray',
               alpha=0.3)

# Customizar o título e labels
plt.title('Distribuição dos Valores Totais Homologados', 
          pad=20, 
          fontsize=14, 
          fontweight='bold')
plt.ylabel('Valor Total Homologado (R$)', 
          fontsize=12)

# Formatar os valores do eixo y para melhor legibilidade
def format_func(value, tick_number):
    return f'R$ {value:,.0f}'

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

# Rotacionar os labels do eixo y para melhor legibilidade
plt.xticks(rotation=0)

# Adicionar grid
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Calcular e adicionar estatísticas básicas como texto
estatisticas = dados['valorTotalHomologado'].describe()
stats_text = f'Média: R$ {estatisticas["mean"]:,.2f}\n'
stats_text += f'Mediana: R$ {estatisticas["50%"]:,.2f}\n'
stats_text += f'Desvio Padrão: R$ {estatisticas["std"]:,.2f}'

# Adicionar texto com estatísticas
plt.text(1.15, 0.5, stats_text,
         transform=ax.transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         verticalalignment='center')

# Ajustar os limites do gráfico para acomodar o texto
plt.subplots_adjust(right=0.85)

# Adicionar título ao eixo x
plt.xlabel('Distribuição', fontsize=12)

# Remover os ticks do eixo x já que temos apenas uma variável
plt.xticks([])

plt.tight_layout()
plt.show()

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configurações gerais do plot
plt.style.use('default')  # Usando estilo default do matplotlib
sns.set_theme(style="whitegrid")  # Aplicando tema do seaborn
sns.set_palette("husl")

# Criar figura com tamanho personalizado
plt.figure(figsize=(10, 6))

# Criar histograma da coluna valorTotalHomologado
sns.histplot(dados['valorTotalHomologado'], bins=15, kde=True, color='skyblue', alpha=0.7)

# Customizar o título e labels
plt.title('Distribuição de Valores Totais Homologados', fontsize=16, fontweight='bold')
plt.xlabel('Valor Total Homologado (R$)', fontsize=14)
plt.ylabel('Frequência', fontsize=14)

# Formatar os valores do eixo x para melhor legibilidade
def format_func(value, tick_number):
    return f'R$ {value:,.0f}'

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))

# Adicionar grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajustar os limites do gráfico
plt.xlim(dados['valorTotalHomologado'].min(), dados['valorTotalHomologado'].max())

# Exibir o gráfico
plt.tight_layout()
plt.show()

# %%
#considerando a alta correlação e observando que valor estimado e valor homolado são iguais em todas ocorrencias
#exceto quando o valorTotalHomolagado está ausente, optaremos pela remoção do valor totalHomologado, que possui 326 dados ausentes
#%%

dados = dados.drop(columns=["valorTotalHomologado","diff_homolagado_estimado"])
#  Identificadores
#  1   modalidadeId                       2967 non-null   int64  
#  23  modoDisputaNome                    2967 non-null   object 

# reprensenta um identificador do tipo de modalidade, no caso 8, que representa a dispensa de licitação
dados["modalidadeId"].unique()
dados["modalidadeNome"].unique()
#sendo assim, eliminaremos essas colunas
dados.drop(columns = ["modalidadeId", "modalidadeNome"], inplace=True)
#
#  2   srp                                2967 non-null   bool  
#representa quando as dispensas são do tipo registro de preço
dados["srp"].value_counts() 

# Analisando colunas nulas
colunas_para_remover = dados.isnull().all().index[dados.isnull().all()].tolist()
print(f'colunas nulas {colunas_para_remover}')

#['orgaoSubRogado', 'justificativaPresencial', 'unidadeSubRogada', 
# 'linkProcessoEletronico']

dados = dados.drop(columns = colunas_para_remover)

# Verificar colunas com apenas um dado não nulo
colunas_para_remover_um_dado = dados.columns[dados.notnull().sum() == 1]
#['orgaoSubRogado.razaoSocial', 'orgaoSubRogado.poderId',
#'orgaoSubRogado.esferaId', 'orgaoSubRogado.cnpj'
#'unidadeSubRogada.ufNome', 'unidadeSubRogada.codigoUnidade',
#'unidadeSubRogada.nomeUnidade', 'unidadeSubRogada.ufSigla',
#'unidadeSubRogada.municipioNome', 'unidadeSubRogada.codigoIbge']

# Remover as colunas que têm apenas um dado não nulo
dados = dados.drop(columns=colunas_para_remover_um_dado)

# remoção de colunas redundantes ou denecessárias para análise
# sequencialCompra: Sequencial da Contratação no PNCP; número sequencial gerado no momento da inclusão.
# informacaoComplementar: Informação complementar sobre o objeto referente à contratação.
# processo: Número do processo de contratação no sistema de origem.
# objetoCompra: Descrição do objeto referente à contratação, detalhando o que está sendo adquirido.
# linkSistemaOrigem: URL para a página/portal do sistema de origem da contratação para recebimento de propostas.
# numeroCompra: Número da contratação no sistema de origem.
# numeroControlePNCP: Número de controle PNCP da contratação, que identifica unicamente o registro no sistema.
# modoDisputaNome: Nome do modo de disputa referente à contratação, que pode indicar como as propostas são apresentadas e avaliadas.
# tipoInstrumentoConvocatorioNome: Nome do instrumento convocatório da contratação, que pode ser um edital ou outro documento formal.
# situacaoCompraNome: Situação atual da contratação, que pode indicar se está em andamento, encerrada, etc.
# usuarioNome: Nome do usuário ou sistema que enviou a contratação, ajudando a identificar o responsável pela inclusão.
# orgaoEntidade.cnpj: CNPJ do órgão referente à contratação, que é o identificador tributário da entidade pública.
# orgaoEntidade.razaoSocial: Razão social do órgão referente à contratação, ou seja, o nome jurídico da entidade.
# amparoLegal.descricao: Descrição do amparo legal que fundamenta a contratação, podendo incluir a legislação ou norma relevante.
# amparoLegal.nome: Nome do amparo legal que justifica a contratação, referindo-se à norma ou legislação específica.
# unidadeOrgao.ufNome: Nome da unidade federativa (estado) onde está localizada a unidade administrativa do órgão.
# unidadeOrgao.municipioNome: Nome do município onde está situada a unidade administrativa do órgão.
# unidadeOrgao.nomeUnidade: Nome da unidade administrativa pertencente ao órgão, identificando a unidade responsável pela contratação.

colunas_a_remover = [
    'sequencialCompra',
    'informacaoComplementar',
    'processo',
    'objetoCompra',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modoDisputaNome',
    'tipoInstrumentoConvocatorioNome',
    'situacaoCompraNome',
    'usuarioNome',
    'orgaoEntidade.cnpj',
    'orgaoEntidade.razaoSocial',
    'amparoLegal.descricao',
    'amparoLegal.nome',
    'unidadeOrgao.ufNome',
    'unidadeOrgao.municipioNome',
    'unidadeOrgao.nomeUnidade',
    'unidadeOrgao.codigoUnidade'
]

# Remover as colunas
dados.drop(columns=colunas_a_remover, inplace=True)

#  Analisando as colunas de datas.

from itertools import combinations

# Converter as colunas para o formato de data
date_columns = ['dataAberturaProposta', 
                'dataEncerramentoProposta', 
                'dataInclusao', 
                'dataPublicacaoPncp', 
                'dataAtualizacao']
for col in date_columns:
    dados[col] = pd.to_datetime(dados[col])

# Comparar colunas dois a dois
comparison_results = {}

for col1, col2 in combinations(date_columns, 2):
    # Obter valores únicos de cada coluna
    unique_values_col1 = set(dados[col1].unique())
    unique_values_col2 = set(dados[col2].unique())
    
    # Comparar os conjuntos
    are_equal = unique_values_col1 == unique_values_col2
    comparison_results[(col1, col2)] = are_equal

# Exibir os resultados
for (col1, col2), are_equal in comparison_results.items():
    print(f"As colunas '{col1}' e '{col2}' têm os mesmos valores? {'Sim' if are_equal else 'Não'}")

# As colunas 'dataAberturaProposta' e 'dataEncerramentoProposta' têm os mesmos valores? Não
# As colunas 'dataAberturaProposta' e 'dataInclusao' têm os mesmos valores? Não
# As colunas 'dataAberturaProposta' e 'dataPublicacaoPncp' têm os mesmos valores? Não
# As colunas 'dataAberturaProposta' e 'dataAtualizacao' têm os mesmos valores? Não
# As colunas 'dataEncerramentoProposta' e 'dataInclusao' têm os mesmos valores? Não
# As colunas 'dataEncerramentoProposta' e 'dataPublicacaoPncp' têm os mesmos valores? Não
# As colunas 'dataEncerramentoProposta' e 'dataAtualizacao' têm os mesmos valores? Não
# As colunas 'dataInclusao' e 'dataPublicacaoPncp' têm os mesmos valores? Sim
# As colunas 'dataInclusao' e 'dataAtualizacao' têm os mesmos valores? Não
# As colunas 'dataPublicacaoPncp' e 'dataAtualizacao' têm os mesmos valores? Não

dados.drop(columns="dataPublicacaoPncp", inplace=True)

#  Criando features que podem ser uteis

# Calcular a duração da proposta
# Para garantir que não haja erros, vamos tratar os casos em que os dados podem estar ausentes
dados['duracaoProposta'] = (dados['dataEncerramentoProposta'] - dados['dataAberturaProposta']).dt.days

# Calcular a duração entre inclusão e atualização
dados['duracaoInclusaoAtualizacao'] = (dados['dataAtualizacao'] - dados['dataInclusao']).dt.days

dados[["duracaoInclusaoAtualizacao","duracaoProposta"]]

# Calcular a média das durações
media_duracao_proposta = dados['duracaoProposta'].mean()
media_duracao_inclusao_atualizacao = dados['duracaoInclusaoAtualizacao'].mean()

# Criar um DataFrame para facilitar a plotagem
duracoes = pd.DataFrame({
    'Tipo': ['Duração da Proposta', 'Duração entre Inclusão e Atualização'],
    'Média em Dias': [media_duracao_proposta, media_duracao_inclusao_atualizacao]
})

# Criar o gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(duracoes['Tipo'], duracoes['Média em Dias'], color=['skyblue'])
plt.title('Comparação das Durações')
plt.xlabel('Tipo de Duração')
plt.ylabel('Média em Dias')
plt.grid(axis='y')

# Exibir o gráfico
plt.tight_layout()
plt.show()

colunas = ["duracaoInclusaoAtualizacao","duracaoProposta"]

# trabalhando as categóricas:
categoricas = dados.select_dtypes(include='object')
lista_colunas_categoricas = categoricas.columns.tolist()

# Remover a coluna 'unidadeOrgao.codigoUnidade'
lista_colunas_categoricas = [col for col in lista_colunas_categoricas if col != 'unidadeOrgao.codigoUnidade']

# Aplicar One-Hot Encoding
dados = pd.get_dummies(dados, columns=lista_colunas_categoricas, drop_first=True)

def analyze_correlations(dados, num_correlacoes=10, figsize_heatmap=(50,30), figsize_barplot=(12,8)):
    """
    Realiza análise completa de correlações em um DataFrame, gerando heatmap e gráfico de barras
    das maiores e menores correlações.
    
    Parâmetros:
    dados (pd.DataFrame): DataFrame para análise
    num_correlacoes (int): Número de maiores/menores correlações a serem exibidas
    figsize_heatmap (tuple): Tamanho da figura do heatmap
    figsize_barplot (tuple): Tamanho da figura do gráfico de barras
    
    Retorna:
    tuple: (correlacao, correlacoes_ordenadas, maiores_correlacoes, menores_correlacoes)
    """
    # Calcula a matriz de correlação
    correlacao = dados.corr()
    
    # Remove correlações duplicadas e auto-correlações
    mask = np.tril(np.ones(correlacao.shape)).astype(bool)
    correlacao_sem_diag = correlacao.where(~mask).stack()
    
    # Ordena correlações pelo valor absoluto
    correlacoes_ordenadas = correlacao_sem_diag.abs().sort_values(ascending=False)
    
    # Seleciona as maiores e menores correlações
    maiores_correlacoes = correlacoes_ordenadas.head(num_correlacoes)
    menores_correlacoes = correlacoes_ordenadas.tail(num_correlacoes)
    
    def plot_heatmap():
        """Gera o heatmap de correlação"""
        plt.figure(figsize=figsize_heatmap)
        sns.heatmap(correlacao, 
                   annot=True, 
                   cmap='coolwarm', 
                   linewidths=0.5,
                   fmt='.2f')
        plt.title('Matriz de Correlação')
        plt.tight_layout()
        plt.show()
    
    def plot_top_correlations():
        """Gera o gráfico de barras das maiores/menores correlações"""
        # Prepara os dados para o gráfico
        correlacoes_para_plotar = pd.concat([maiores_correlacoes, menores_correlacoes])
        
        # Cria labels mais legíveis para o eixo y
        labels = [f"{idx[0]} → {idx[1]}" for idx in correlacoes_para_plotar.index]
        valores = correlacoes_para_plotar.values
        
        plt.figure(figsize=figsize_barplot)
        
        # Cria o gráfico de barras
        bars = plt.barh(range(len(labels)), valores)
        
        # Customiza as cores baseado nos valores
        for i, bar in enumerate(bars):
            if valores[i] >= 0:
                bar.set_color('#d65f5f')  # Vermelho para correlações positivas
            else:
                bar.set_color('#5fba7d')  # Verde para correlações negativas
        
        # Customiza o gráfico
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Correlação')
        plt.title(f'Top {num_correlacoes} Maiores e Menores Correlações')
        
        # Adiciona os valores nas barras
        for i, v in enumerate(valores):
            plt.text(v + (0.01 if v >= 0 else -0.01), 
                    i,
                    f'{v:.2f}',
                    va='center',
                    ha='left' if v >= 0 else 'right')
        
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    
    # Exibe os resultados
    print("\nMaiores correlações:")
    for idx, valor in maiores_correlacoes.items():
        print(f"{idx[0]} → {idx[1]}: {valor:.3f}")
    
    print("\nMenores correlações:")
    for idx, valor in menores_correlacoes.items():
        print(f"{idx[0]} → {idx[1]}: {valor:.3f}")
    
    # Gera os gráficos
    plot_heatmap()
    plot_top_correlations()
    
    return correlacao, correlacoes_ordenadas, maiores_correlacoes, menores_correlacoes


resultados = analyze_correlations(dados, num_correlacoes=10)
correlacao, correlacoes_ordenadas, maiores_correlacoes, menores_correlacoes = resultados

# %% considerando as correlações, os seguintes dados serão suprimidos por apresentarem o mesmo comportamento sempre. 
# modoDisputaId → tipoInstrumentoConvocatorioCodigo: 1.000
# dataAberturaProposta → dataEncerramentoProposta: 1.000
# dataAberturaProposta → tipoInstrumentoConvocatorioCodigo: 1.000
# dataAberturaProposta → modoDisputaId: 1.000
# dataEncerramentoProposta → modoDisputaId: 1.000
# dataEncerramentoProposta → tipoInstrumentoConvocatorioCodigo: 1.000

colunas_alta_correlacao =[
    'tipoInstrumentoConvocatorioCodigo',
]

dados = dados.drop(columns = colunas_alta_correlacao)


# %%
dados.to_pickle('dados/processados/Eda.pkl')

################################ Iniciando Mineração de Dados #############
# %%
df = pd.read_pickle('dados/processados/eda.pkl')

# Excluindo colunas do tipo data
df = dados.select_dtypes(exclude=['datetime64[ns]'])

# %%Imputar valor 0 para os valores ausentes na coluna duracaoProposta
df['duracaoProposta'].fillna(0, inplace=True)

# %% Converter booleanos em int
df = df.astype(int)

# %%
from sklearn.preprocessing import  StandardScaler

scaler = StandardScaler()
df_normalizados = scaler.fit_transform(df)
df_normalizados = pd.DataFrame(df_normalizados, columns=df.columns)

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

# Exemplo de uso:
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

# %%

df_normalizados = scaler.fit_transform(df)

# Executar análise completa
df_results, models, evaluations, best_model = run_anomaly_detection_analysis(
    df, df_normalizados, params
)

# Visualizar resultados
plot_anomalies_comparison(df, df_normalizados)
plot_consensus_anomalies(df, df_normalizados)
print_anomaly_summary(df)


############################### Rede Neural ###################################
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

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


# %%
# Obter as anomalias do melhor modelo tradicional
traditional_anomalies = df_results[f'anomalia_{best_model.lower()}']

# Executar o detector baseado em deep learning
detector, deep_anomalies = run_deep_anomaly_detection(df, traditional_anomalies)

# %%

###################### Analise Final #########################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr;
import matplotlib.gridspec as gridspec
from matplotlib_venn import venn2, venn3

def analyze_consensus_anomalies(df, traditional_results, deep_anomalies, models):
    """
    Analisa as anomalias detectadas em comum entre todos os métodos
    
    Parâmetros:
    df: DataFrame original
    traditional_results: DataFrame com resultados dos métodos tradicionais
    deep_anomalies: array boolean com resultados do deep learning
    models: dicionário com os modelos treinados
    """
    # Criar DataFrame com todos os resultados
    results_df = pd.DataFrame({
        'KMeans': traditional_results['anomalia_kmeans'],
        'DBSCAN': traditional_results['anomalia_dbscan'],
        'LOF': traditional_results['anomalia_lof'],
        'IForest': traditional_results['anomalia_iforest'],
        'DeepLearning': deep_anomalies
    })
    
    # Calcular consenso
    results_df['n_detections'] = results_df.sum(axis=1)
    
    # Identificar anomalias por nível de consenso
    consensus_levels = {
        'full': results_df['n_detections'] == 5,  # Todos os métodos
        'strong': results_df['n_detections'] >= 4, # Pelo menos 4 métodos
        'moderate': results_df['n_detections'] >= 3, # Pelo menos 3 métodos
        'weak': results_df['n_detections'] >= 2  # Pelo menos 2 métodos
    }
    
    # Criar visualizações
    plot_consensus_analysis(df, results_df, consensus_levels)
    
    # Análise detalhada das anomalias em comum
    analyze_common_anomalies(df, results_df, consensus_levels)
    
    return results_df, consensus_levels

def plot_consensus_analysis(df, results_df, consensus_levels):
    """
    Cria visualizações detalhadas da análise de consenso
    """
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3)
    
    # 1. Scatter plot com níveis de consenso
    ax1 = plt.subplot(gs[0, :2])
    scatter = ax1.scatter(df.iloc[:, 0], df.iloc[:, 1],
                         c=results_df['n_detections'],
                         cmap='RdYlBu_r', alpha=0.6)
    ax1.set_title('Níveis de Consenso na Detecção de Anomalias')
    ax1.set_xlabel(df.columns[0])
    ax1.set_ylabel(df.columns[1])
    plt.colorbar(scatter, ax=ax1, label='Número de Métodos que Detectaram')
    
    # 2. Diagrama de Venn (3 principais métodos)
    ax2 = plt.subplot(gs[0, 2])
    top_methods = ['KMeans', 'DBSCAN', 'DeepLearning']
    venn3([set(np.where(results_df[method])[0]) for method in top_methods],
          set_labels=top_methods, ax=ax2)
    ax2.set_title('Interseção entre Principais Métodos')
    
    # 3. Heatmap de correlação entre métodos
    ax3 = plt.subplot(gs[1, 0])
    corr_matrix = results_df.iloc[:, :5].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
    ax3.set_title('Correlação entre Métodos')
    
    # 4. Barplot de contagem por nível de consenso
    ax4 = plt.subplot(gs[1, 1])
    consensus_counts = {
        'Consenso Total (5)': sum(consensus_levels['full']),
        'Forte (≥4)': sum(consensus_levels['strong']),
        'Moderado (≥3)': sum(consensus_levels['moderate']),
        'Fraco (≥2)': sum(consensus_levels['weak'])
    }
    colors = ['darkred', 'red', 'orange', 'yellow']
    ax4.bar(consensus_counts.keys(), consensus_counts.values(), color=colors)
    ax4.set_title('Distribuição dos Níveis de Consenso')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Características das anomalias por consenso
    ax5 = plt.subplot(gs[1, 2])
    boxplot_data = [
        df.iloc[:, 0][consensus_levels['full']],
        df.iloc[:, 0][consensus_levels['strong']],
        df.iloc[:, 0][consensus_levels['moderate']],
        df.iloc[:, 0][consensus_levels['weak']]
    ]
    ax5.boxplot(boxplot_data, labels=['Total', 'Forte', 'Moderado', 'Fraco'])
    ax5.set_title(f'Distribuição de {df.columns[0]} por Nível de Consenso')
    
    # 6. Scatter plots para cada nível de consenso
    consensus_names = ['full', 'strong', 'moderate', 'weak']
    for idx, (name, mask) in enumerate(consensus_levels.items()):
        ax = plt.subplot(gs[2, idx])
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c='lightgray', alpha=0.3)
        ax.scatter(df.iloc[mask, 0], df.iloc[mask, 1], c='red', alpha=0.6)
        ax.set_title(f'Anomalias com Consenso {name.capitalize()}')
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
    
    plt.tight_layout()
    plt.show()

def analyze_common_anomalies(df, results_df, consensus_levels):
    """
    Realiza análise estatística das anomalias em comum
    """
    print("\nAnálise das Anomalias por Nível de Consenso:")
    print("-" * 50)
    
    for level_name, mask in consensus_levels.items():
        n_anomalies = sum(mask)
        print(f"\n{level_name.capitalize()} Consensus ({n_anomalies} anomalias):")
        
        if n_anomalies > 0:
            # Estatísticas básicas
            stats = df[mask].describe()
            print("\nEstatísticas das anomalias:")
            print(stats)
            
            # Características distintivas
            if n_anomalies >= 2:
                for col in df.columns:
                    normal_data = df.loc[~mask, col]
                    anomaly_data = df.loc[mask, col]
                    
                    # Teste de correlação
                    corr, p_value = pearsonr(
                        df[col],
                        results_df['n_detections']
                    )
                    
                    print(f"\nCaracterística: {col}")
                    print(f"Correlação com número de detecções: {corr:.3f} (p-value: {p_value:.3f})")
                    
                    # Comparação de distribuições
                    print(f"Média (Normal): {normal_data.mean():.2f}")
                    print(f"Média (Anomalia): {anomaly_data.mean():.2f}")
                    print(f"Desvio Padrão (Normal): {normal_data.std():.2f}")
                    print(f"Desvio Padrão (Anomalia): {anomaly_data.std():.2f}")

def run_complete_anomaly_analysis(df, df_normalizados):
    """
    Executa análise completa incluindo todos os métodos e consenso
    """
    # 1. Executar métodos tradicionais
    params = {
        'kmeans': {'n_clusters': 2, 'percentil': 95},
        'dbscan': {'eps': 0.5, 'min_samples': 5},
        'lof': {'n_neighbors': 20, 'contamination': 0.1},
        'iforest': {'contamination': 0.1}
    }
    
    traditional_results, models, evaluations, best_model = run_anomaly_detection_analysis(
        df, df_normalizados, params
    )
    
    # 2. Executar deep learning
    detector, deep_anomalies = run_deep_anomaly_detection(
        df, traditional_results[f'anomalia_{best_model.lower()}']
    )
    
    # 3. Análise de consenso
    results_df, consensus_levels = analyze_consensus_anomalies(
        df, traditional_results, deep_anomalies, models
    )
    
    return results_df, consensus_levels, detector, models

# %%
results_df, consensus_levels, detector, models = run_complete_anomaly_analysis(df, df_normalizados)

# %%

consenso = df[(df["anomalia_consenso"] == 3) | (df["anomalia_consenso"] == 4)]
# %%
print(df.describe())

plt.figure(figsize=(16, 16))
df.hist(figsize=(16, 16))
plt.tight_layout()
plt.show()


# %%

# Criar um DataFrame com os resultados de cada método
comparison_df = pd.DataFrame({
    'KMeans': df_results['anomalia_kmeans'],
    'DBSCAN': df_results['anomalia_dbscan'],
    'LOF': df_results['anomalia_lof'],
    'IForest': df_results['anomalia_iforest']
})

# Calcular a porcentagem de anomalias detectadas por cada método
anomaly_percentages = comparison_df.mean() * 100

# Criar um gráfico de barras
plt.figure(figsize=(10, 6))
anomaly_percentages.plot(kind='bar')
plt.title('Porcentagem de Anomalias Detectadas por Método')
plt.ylabel('Porcentagem de Anomalias')
plt.ylim(0, 100)
for i, v in enumerate(anomaly_percentages):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')
plt.tight_layout()
plt.show()

# %%
print(f"Melhor modelo tradicional: {best_model}")
print("\nMétricas de avaliação:")
for metric, value in evaluations[best_model].items():
    print(f"- {metric}: {value:.4f}")

# %%
# Criar um DataFrame com os resultados de todos os métodos
consensus_df = pd.DataFrame({
    'KMeans': df_results['anomalia_kmeans'],
    'DBSCAN': df_results['anomalia_dbscan'],
    'LOF': df_results['anomalia_lof'],
    'IForest': df_results['anomalia_iforest']
})

# Calcular o número de detecções por registro
consensus_df['n_detections'] = consensus_df.sum(axis=1)

# Criar visualização
plt.figure(figsize=(15, 10))

# Subplot 1: Scatter plot do valor total estimado vs duração da proposta
plt.subplot(2, 2, 1)
scatter = plt.scatter(df['valorTotalEstimado'], 
                     df['duracaoProposta'],
                     c=consensus_df['n_detections'],
                     cmap='viridis',
                     alpha=0.6)
plt.colorbar(scatter, label='Número de métodos que detectaram como anomalia')
plt.title('Consenso na Detecção de Anomalias')
plt.xlabel('Valor Total Estimado')
plt.ylabel('Duração da Proposta')

# Subplot 2: Distribuição do número de detecções
plt.subplot(2, 2, 2)
consensus_df['n_detections'].hist(bins=5)
plt.title('Distribuição do Número de Detecções')
plt.xlabel('Número de Métodos')
plt.ylabel('Frequência')

# Definir níveis de consenso
consensus_levels = {
    'Consenso Total (4)': consensus_df['n_detections'] == 4,
    'Forte (≥3)': consensus_df['n_detections'] >= 3,
    'Moderado (≥2)': consensus_df['n_detections'] >= 2,
    'Fraco (≥1)': consensus_df['n_detections'] >= 1
}

# Subplot 3: Barplot do número de anomalias por nível de consenso
plt.subplot(2, 2, 3)
counts = [sum(mask) for mask in consensus_levels.values()]
plt.bar(consensus_levels.keys(), counts)
plt.xticks(rotation=45)
plt.title('Número de Anomalias por Nível de Consenso')
plt.ylabel('Número de Registros')

# Subplot 4: Boxplot dos valores por nível de consenso
plt.subplot(2, 2, 4)
consensus_data = []
labels = []
for level_name, mask in consensus_levels.items():
    consensus_data.append(df.loc[mask, 'valorTotalEstimado'])
    labels.append(level_name)

plt.boxplot(consensus_data, labels=labels)
plt.xticks(rotation=45)
plt.title('Distribuição dos Valores por Nível de Consenso')
plt.ylabel('Valor Total Estimado')

plt.tight_layout()
plt.show()

# Imprimir estatísticas para cada nível de consenso
for level_name, mask in consensus_levels.items():
    n_anomalies = sum(mask)
    print(f"\n{level_name} ({n_anomalies} anomalias):")
    print("\nEstatísticas do Valor Total Estimado:")
    print(df.loc[mask, 'valorTotalEstimado'].describe())
    print("\nEstatísticas da Duração da Proposta:")
    print(df.loc[mask, 'duracaoProposta'].describe())
    print("-" * 50)
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assumindo que 'df' é o seu DataFrame original e 'df_results' contém os resultados dos métodos de detecção de anomalias

# Preparar os dados para PCA
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Aplicar PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Criar um DataFrame com os resultados de todos os métodos
consensus_df = pd.DataFrame({
    'KMeans': df_results['anomalia_kmeans'],
    'DBSCAN': df_results['anomalia_dbscan'],
    'LOF': df_results['anomalia_lof'],
    'IForest': df_results['anomalia_iforest']
})

# Calcular o número de detecções por registro
consensus_df['n_detections'] = consensus_df.sum(axis=1)

# Criar visualização
plt.figure(figsize=(20, 15))

# Subplot 1: Scatter plot do PCA com consenso
plt.subplot(2, 2, 1)
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1],
                     c=consensus_df['n_detections'],
                     cmap='viridis',
                     alpha=0.6)
plt.colorbar(scatter, label='Número de métodos que detectaram como anomalia')
plt.title('Consenso na Detecção de Anomalias (PCA)')
plt.xlabel('Primeira Componente Principal')
plt.ylabel('Segunda Componente Principal')

# Subplot 2: Distribuição do número de detecções
plt.subplot(2, 2, 2)
consensus_df['n_detections'].hist(bins=5)
plt.title('Distribuição do Número de Detecções')
plt.xlabel('Número de Métodos')
plt.ylabel('Frequência')

# Definir níveis de consenso
consensus_levels = {
    'Consenso Total (4)': consensus_df['n_detections'] == 4,
    'Forte (≥3)': consensus_df['n_detections'] >= 3,
    'Moderado (≥2)': consensus_df['n_detections'] >= 2,
    'Fraco (≥1)': consensus_df['n_detections'] >= 1
}

# Subplot 3: Barplot do número de anomalias por nível de consenso
plt.subplot(2, 2, 3)
counts = [sum(mask) for mask in consensus_levels.values()]
plt.bar(consensus_levels.keys(), counts)
plt.xticks(rotation=45)
plt.title('Número de Anomalias por Nível de Consenso')
plt.ylabel('Número de Registros')

# Subplot 4: Boxplot dos valores da primeira componente principal por nível de consenso
plt.subplot(2, 2, 4)
consensus_data = []
labels = []
for level_name, mask in consensus_levels.items():
    consensus_data.append(df_pca[mask, 0])
    labels.append(level_name)

plt.boxplot(consensus_data, labels=labels)
plt.xticks(rotation=45)
plt.title('Distribuição da Primeira Componente Principal por Nível de Consenso')
plt.ylabel('Valor da Primeira Componente Principal')

plt.tight_layout()
plt.show()

# Imprimir estatísticas para cada nível de consenso
for level_name, mask in consensus_levels.items():
    n_anomalies = sum(mask)
    print(f"\n {level_name} ({n_anomalies} anomalias):")
    print("\nEstatísticas da Primeira Componente Principal:")
    print(pd.Series(df_pca[mask, 0]).describe())
    print("\nEstatísticas da Segunda Componente Principal:")
    print(pd.Series(df_pca[mask, 1]).describe())
    print("-" * 50)

# %%
