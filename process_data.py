# %%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# %% Leitura dos dados
dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv', index_col='Unnamed: 0')
# %% Remover colunas com todos os valores nulos ou apenas um valor não nulo
# Criando a máscara
mascara = dados.isnull().sum() >= 2966
# Filtrando apenas as colunas com valor True
colunas_nulas = mascara[mascara].index.tolist()
dados = dados.drop(columns=colunas_nulas)
#%% Remover colunas redundantes
# A coluna 'modalidadeId' possui o valor fixo 8, indicando que todas as linhas correspondem à modalidade de dispensa de licitações, portanto, não é relevante para a análise.
# As colunas 'modalidadeNome', 'situacaoCompraNome', 'usuarioNome', e 'orgaoEntidade.razaoSocial' são redundantes, pois já possuímos o 'modalidadeId' e 'situacaoCompraId' com a mesma informação em formato numérico.
# Outras colunas, como 'informacaoComplementar', 'objetoCompra', 'linkSistemaOrigem', e detalhes sobre localização como 'unidadeOrgao.ufNome', 'unidadeOrgao.ufSigla', entre outras, não são necessárias para a análise pretendida.

colunas_para_remover = [
    'modalidadeId', 'modalidadeNome', 'situacaoCompraNome', 'usuarioNome', 
    'orgaoEntidade.razaoSocial', 'informacaoComplementar', 'objetoCompra', 
    'linkSistemaOrigem', 'unidadeOrgao.ufNome', 'unidadeOrgao.ufSigla',
    'amparoLegal.descricao', 'amparoLegal.nome', 'unidadeOrgao.municipioNome', 
    'unidadeOrgao.nomeUnidade','tipoInstrumentoConvocatorioNome', 'modoDisputaNome', 
    'orgaoEntidade.cnpj','numeroCompra','numeroControlePNCP'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_para_remover)

#%% Analisando as datas
# Converter colunas de data
colunas_data = ['dataAberturaProposta', 'dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')

# Calcular a diferença entre as datas
dados['diff_abertura_encerramento'] = (dados['dataEncerramentoProposta'] - dados['dataAberturaProposta']).dt.days
dados['diff_inclusao_atualizacao'] = (dados['dataAtualizacao'] - dados['dataInclusao']).dt.days
# Converter datas em número de dias desde a data mínima
min_date = dados[['dataAberturaProposta', 'dataEncerramentoProposta', 'dataInclusao', 'dataAtualizacao']].min().min()

dados['days_from_min_abertura'] = (dados['dataAberturaProposta'] - min_date).dt.days
dados['days_from_min_encerramento'] = (dados['dataEncerramentoProposta'] - min_date).dt.days
dados['days_from_min_inclusao'] = (dados['dataInclusao'] - min_date).dt.days
dados['days_from_min_atualizacao'] = (dados['dataAtualizacao'] - min_date).dt.days

# Calcular a correlação entre essas colunas
correlacao = dados[['days_from_min_abertura', 'days_from_min_encerramento', 'days_from_min_inclusao', 'days_from_min_atualizacao']].corr()

# Comparar se os valores ausentes ocorrem nas mesmas linhas para ambas as colunas
ausentes_mesmas_linhas = dados['dataAberturaProposta'].isnull() == dados['dataEncerramentoProposta'].isnull()

# Verificar se todos os valores são True, o que indicaria que os dados ausentes ocorrem nas mesmas linhas
mesmos_ausentes = ausentes_mesmas_linhas.all()

# Exibir o resultado
if mesmos_ausentes:
    print("Os dados ausentes ocorrem nas mesmas linhas para as duas variáveis.")
else:
    print("Os dados ausentes não ocorrem nas mesmas linhas para as duas variáveis.")


#exluindo uma das datas com maior correlação 0.999129 entre dataAberturaProposta e dataEncerramentoProposta  
dados = dados.drop(columns = ['diff_abertura_encerramento',
       'diff_inclusao_atualizacao', 'days_from_min_abertura',
       'days_from_min_encerramento', 'days_from_min_inclusao',
       'days_from_min_atualizacao', 'dataEncerramentoProposta'])

# %% Observando a correlação
# Seleciona apenas colunas numéricas
dados_numericos = dados.select_dtypes(include=['float64', 'int64'])
# Calcula a matriz de correlação
matriz_correlacao = dados_numericos.corr()

#considerando a alta correlação entre ValorTotalHomologado e ValotTotalEstimado, opta-se pela eliminação do valorTotalEstimado, evitando a redundância
dados = dados.drop(columns='valorTotalEstimado')

matriz_correlacao

# %% Imputando valores faltantes em valorTotalHomologado
# # Cálculo da mediana e tratamento de valores ausentes


# Cálculo dos quartis e do IQR (Interquartile Range)
q1 = dados['valorTotalHomologado'].quantile(0.25)
q3 = dados['valorTotalHomologado'].quantile(0.75)
iqr = q3 - q1

# Definindo limites para outliers
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

# Filtrando os outliers
outliers = dados[(dados['valorTotalHomologado'] < limite_inferior) | 
                 (dados['valorTotalHomologado'] > limite_superior)]

# Exibindo o número de outliers
num_outliers = outliers.shape[0]
print(f'Número de outliers em valorTotalHomologado: {num_outliers}')

mediana_valorTotal = dados['valorTotalHomologado'].median()
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)


# %% One-hot encoding para as colunas categóricas específicas
colunas_one_hot = [
    'modoDisputaId', 
    'situacaoCompraId', 
    'tipoInstrumentoConvocatorioCodigo', 
    'amparoLegal.codigo', 
    'orgaoEntidade.poderId', 
    'orgaoEntidade.esferaId'
]

for coluna in colunas_one_hot:
    # Criar dummies
    dummies = pd.get_dummies(dados[coluna], prefix=coluna)
    
    # Garantir que todas as colunas necessárias existam
    if coluna == 'modoDisputaId':
        for i in [4, 5]:
            col_name = f'{coluna}_{i}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0
    elif coluna == 'situacaoCompraId':
        for i in [1, 2, 3]:
            col_name = f'{coluna}_{i}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0
    elif coluna == 'tipoInstrumentoConvocatorioCodigo':
        for i in [2, 3]:
            col_name = f'{coluna}_{i}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0
    elif coluna == 'amparoLegal.codigo':
        for i in [18, 19, 20, 21, 22, 24, 36, 37, 38, 39, 41, 45]:
            col_name = f'{coluna}_{i}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0
    elif coluna == 'orgaoEntidade.poderId':
        for valor in ['E', 'N', 'L', 'J']:
            col_name = f'{coluna}_{valor}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0
    elif coluna == 'orgaoEntidade.esferaId':
        for valor in ['F', 'M', 'E', 'N', 'D']:
            col_name = f'{coluna}_{valor}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0

    # Adicionar as dummies ao dataframe
    dados = pd.concat([dados, dummies], axis=1)

#categoria é substituída pela frequência ou contagem de ocorrências dessa categoria no conjunto de dados.
freq_encoding = dados['unidadeOrgao.codigoIbge'].value_counts()
dados['unidadeOrgao.codigoIbge'] = dados['unidadeOrgao.codigoIbge'].map(freq_encoding)
freq_encoding = dados['unidadeOrgao.codigoUnidade'].value_counts()
dados['unidadeOrgao.codigoUnidade'] = dados['unidadeOrgao.codigoUnidade'].map(freq_encoding)


    
# Remover as colunas originais que foram transformadas
dados = dados.drop(columns=colunas_one_hot)

#%% Remover duplicatas
dados = dados.drop_duplicates()

# %% Selecionar apenas as colunas numéricas para normalização
dados_numericos = dados.select_dtypes(include=['float64', 'int64']).columns

# Instanciar o StandardScaler e aplicar a normalização
scaler = StandardScaler()
dados[dados_numericos] = scaler.fit_transform(dados[dados_numericos])

# %% Salvar o DataFrame normalizado em um novo arquivo CSV
#dados.to_csv('dados_processados.csv', index=False)

# %%
dados