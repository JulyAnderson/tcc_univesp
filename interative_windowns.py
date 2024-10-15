# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

# %%
dados

# %%
dados.columns

# %%
len(dados.columns)

# %%
dados.columns.unique()

# %%
dados.info()

# %%
dados.describe()

# %%
colunas_nulas = dados.columns[dados.isnull().all()]

# %%
colunas_nulas

# %%
dados = dados.drop(columns=colunas_nulas)


# %%
print(dados.dtypes)

# %%
tipos_de_dados= dados.dtypes
tipos_de_dados

# %%
print(tipos_de_dados)

# %%
print(dados.describe(include='object'))  # Para colunas categóricas

# %%
dados.describe(include='object') # Para colunas categóricas

# %%
categoricas = dados.describe(include='object') # Para colunas categóricas

# %%
categoricas

# %%
dados.dataAberturaProposta

# %%
dados['data de abertura'] = pd.to_datetime(dados['data de abertura'], errors='coerce')

# %%
dados['dataAberturaProposta'] = pd.to_datetime(dados['dataAberturaProposta'], errors='coerce')

# %%
dados['dataAberturaProposta'] 

# %%
dtypes(dados['dataAberturaProposta'] )

# %%
dados['dataAberturaProposta'].dtypes()

# %%
dados.columns

# %%
(len(dados.columns))

# %%
categoricas = dados(include='object') # Para colunas categóricas

# %%
categoricas = dados.select_dtypes(include='object')

# %%
categoricas.columns

# %%
dados['dataEncemanetoProposta','dataInclusao','dataPublicacaoPncp', 'dataAtualizacao']

# %%
dados[dados['dataEncemanetoProposta','dataInclusao','dataPublicacaoPncp', 'dataAtualizacao']]

# %%
dados[['dataEncemanetoProposta','dataInclusao','dataPublicacaoPncp', 'dataAtualizacao']]

# %%
dados[['dataEncaminhamentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']]


# %%
dados[['dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']]


# %%
# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')



# %%
categoricas = dados.select_dtypes(include='object')

# %%
categiricas

# %%
categoricas

# %%
categoricas.columns

# %%
missing_values = dados[colunas_categoricas].isnull().sum()

# %%
missing_values = dados[categoricas].isnull().sum()


# %%
missing_values = dados[categoricas.columns].isnull().sum()


# %%
missing_values

# %%
dados.shape()

# %%
dados.shape

# %%
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

# %%
dados.info()

# %%
categoricas

# %%
categoricas.columns

# %%
dados = dados.drop_duplicates()

# %%
dados.shape()

# %%
dados.shape

# %%
dados_encoded = pd.get_dummies(dados[colunas_categoricas], drop_first=True)

# %%
dados_encoded = pd.get_dummies(dados[categoricas], drop_first=True)


# %%
dados_encoded = pd.get_dummies(dados[categoricas.columns], drop_first=True)


# %%
categoricas

# %%
categoricas.columns

# %%
categoricas.columns

# %%
categorias.head()

# %%
categoricas.head()

# %%
categoricas.info()

# %%
# Removendo as colunas com poucos dados não nulos
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 'orgaoSubRogado.poderId', 'orgaoSubRogado.esferaId',
    'unidadeSubRogada.ufNome', 'unidadeSubRogada.nomeUnidade',
    'unidadeSubRogada.ufSigla', 'unidadeSubRogada.municipioNome'
]
categoricas = categoricas.drop(columns=colunas_a_remover)


# %%
categoricas.head()

# %%
categoricas['informacaoComplementar'].fillna("Sem informação", inplace=True)
categoricas['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)


# %%
categoricas.head()

# %%
dados.info()

# %%
import pandas as pd 
dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')
#filtrando as variáveis com todos os valores nulos

colunas_nulas = dados.columns[dados.isnull().all()]
dados = dados.drop(columns=colunas_nulas)
# Lista das colunas a serem convertidas para datetime

colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']
# Converte as colunas especificadas para o tipo datetime

for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')
#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover

colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome'
]
# Removendo as colunas do DataFrame

dados = dados.drop(columns=colunas_a_remover)
#Preenchendo informações faltantes. 

dados['informacaoComplementar'].fillna("Sem informação", inplace=True)
dados['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)

# %%
dados.info()

# %%
dados['valorTotalHomologado'].describe()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma
plt.figure(figsize=(10, 5))
sns.histplot(dados['valorTotalHomologado'], bins=30, kde=True)
plt.title('Distribuição de valorTotalHomologado')
plt.xlabel('valorTotalHomologado')
plt.ylabel('Frequência')
plt.show()

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=dados['valorTotalHomologado'])
plt.title('Boxplot de valorTotalHomologado')
plt.xlabel('valorTotalHomologado')
plt.show()

# %%
!pip install matplotlib

# %%
Q1 = dados['valorTotalHomologado'].quantile(0.25)
Q3 = dados['valorTotalHomologado'].quantile(0.75)
IQR = Q3 - Q1
lim_inferior = Q1 - 1.5 * IQR
lim_superior = Q3 + 1.5 * IQR

outliers = dados[(dados['valorTotalHomologado'] < lim_inferior) | (dados['valorTotalHomologado'] > lim_superior)]
print(f'Número de outliers: {outliers.shape[0]}')


# %%
mport matplotlib.pyplot as plt
import seaborn as sns

# Histograma
plt.figure(figsize=(10, 5))
sns.histplot(dados['valorTotalHomologado'], bins=30, kde=True)
plt.title('Distribuição de valorTotalHomologado')
plt.xlabel('valorTotalHomologado')
plt.ylabel('Frequência')
plt.show()

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=dados['valorTotalHomologado'])
plt.title('Boxplot de valorTotalHomologado')
plt.xlabel('valorTotalHomologado')
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma
plt.figure(figsize=(10, 5))
sns.histplot(dados['valorTotalHomologado'], bins=30, kde=True)
plt.title('Distribuição de valorTotalHomologado')
plt.xlabel('valorTotalHomologado')
plt.ylabel('Frequência')
plt.show()

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=dados['valorTotalHomologado'])
plt.title('Boxplot de valorTotalHomologado')
plt.xlabel('valorTotalHomologado')
plt.show()

# %%
!pip install seaborn


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma
plt.figure(figsize=(10, 5))
sns.histplot(dados['valorTotalHomologado'], bins=30, kde=True)
plt.title('Distribuição de valorTotalHomologado')
plt.xlabel('valorTotalHomologado')
plt.ylabel('Frequência')
plt.show()

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=dados['valorTotalHomologado'])
plt.title('Boxplot de valorTotalHomologado')
plt.xlabel('valorTotalHomologado')
plt.show()

# %%
dados['valorTotalHomologado'].describe()

# %%
mediana_valorTotal = dados['valorTotalHomologado'].median()
# Imputar os valores ausentes com a mediana

dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)

# %%
dados.info()

# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

#filtrando as variáveis com todos os valores nulos
colunas_nulas = dados.columns[dados.isnull().all()]

dados = dados.drop(columns=colunas_nulas)

# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')


#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

#Preenchendo informações faltantes. 
dados['informacaoComplementar'].fillna("Sem informação", inplace=True)
dados['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)

#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'
mediana_valorTotal = dados['valorTotalHomologado'].median()

# Imputar os valores ausentes com a mediana
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)

# %%
dados.info()

# %%
# Matriz de correlação
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# %%
# Matriz de correlação
correlation_matrix = dados.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# %%
categoricas = dados.select_dtypes(include='object')

# %%
categoricas

# %%
categoricas.columns

# %%


# Colunas a serem excluídas
colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modoDisputaNome',
    'usuarioNome',
    'orgaoEntidade.poderId',
    'orgaoEntidade.esferaId',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
    'unidadeOrgao.codigoUnidade'
]

# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

# Visualizar as colunas restantes
print("Colunas após exclusão:")
print(df_reduzido.columns)

# %%
len(df_reduzido)

# %%
len(df_reduzido.columns)

# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

#filtrando as variáveis com todos os valores nulos
colunas_nulas = dados.columns[dados.isnull().all()]

dados = dados.drop(columns=colunas_nulas)

# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')


#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

#Preenchendo informações faltantes. 
dados['informacaoComplementar'].fillna("Sem informação", inplace=True)
dados['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)

#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'
mediana_valorTotal = dados['valorTotalHomologado'].median()

# Imputar os valores ausentes com a mediana
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)


# Colunas a serem excluídas
colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]

# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.

# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

# %%
df

# %%
df_reduzido

# %%
df_reduzido.columns

# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

#filtrando as variáveis com todos os valores nulos
colunas_nulas = dados.columns[dados.isnull().all()]

dados = dados.drop(columns=colunas_nulas)

# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')


#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

#Preenchendo informações faltantes. 
dados['informacaoComplementar'].fillna("Sem informação", inplace=True)
dados['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)

#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'
mediana_valorTotal = dados['valorTotalHomologado'].median()

# Imputar os valores ausentes com a mediana
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)


# Colunas a serem excluídas
colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modelaideId',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]

# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.

# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

colunas_a_excluir = [
    ['tipoInstrumentoConvocatorioNome',
        'situacaoCompraId',
        'orgaoEntidade.cnpj',
       'orgaoEntidade.poderId', 
       'orgaoEntidade.esferaId', 
       'amparoLegal.codigo', 
       'unidadeOrgao.ufNome',
     'unidadeOrgao.codigoUnidade',
     'unidadeOrgao.codigoIbge']
]

#As colunas acima são reduntantes, pois todas apresentam uma coluna correspondente no formato int que representa o id da informação.
# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

#filtrando as variáveis com todos os valores nulos
colunas_nulas = dados.columns[dados.isnull().all()]

dados = dados.drop(columns=colunas_nulas)

# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')


#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

#Preenchendo informações faltantes. 
dados['informacaoComplementar'].fillna("Sem informação", inplace=True)
dados['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)

#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'
mediana_valorTotal = dados['valorTotalHomologado'].median()

# Imputar os valores ausentes com a mediana
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)


# Colunas a serem excluídas
colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modelideId',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]

# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.

# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

colunas_a_excluir = [
    ['tipoInstrumentoConvocatorioNome',
        'situacaoCompraId',
        'orgaoEntidade.cnpj',
       'orgaoEntidade.poderId', 
       'orgaoEntidade.esferaId', 
       'amparoLegal.codigo', 
       'unidadeOrgao.ufNome',
     'unidadeOrgao.codigoUnidade',
     'unidadeOrgao.codigoIbge']
]

#As colunas acima são reduntantes, pois todas apresentam uma coluna correspondente no formato int que representa o id da informação.
# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

#filtrando as variáveis com todos os valores nulos
colunas_nulas = dados.columns[dados.isnull().all()]

dados = dados.drop(columns=colunas_nulas)

# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')


#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

#Preenchendo informações faltantes. 
dados['informacaoComplementar'].fillna("Sem informação", inplace=True)
dados['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)

#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'
mediana_valorTotal = dados['valorTotalHomologado'].median()

# Imputar os valores ausentes com a mediana
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)


# Colunas a serem excluídas
colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modelidadeId',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]

# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.

# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

colunas_a_excluir = [
    ['tipoInstrumentoConvocatorioNome',
        'situacaoCompraId',
        'orgaoEntidade.cnpj',
       'orgaoEntidade.poderId', 
       'orgaoEntidade.esferaId', 
       'amparoLegal.codigo', 
       'unidadeOrgao.ufNome',
     'unidadeOrgao.codigoUnidade',
     'unidadeOrgao.codigoIbge']
]

#As colunas acima são reduntantes, pois todas apresentam uma coluna correspondente no formato int que representa o id da informação.
# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

#filtrando as variáveis com todos os valores nulos
colunas_nulas = dados.columns[dados.isnull().all()]

dados = dados.drop(columns=colunas_nulas)

# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')


#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

#Preenchendo informações faltantes. 
dados['informacaoComplementar'].fillna("Sem informação", inplace=True)
dados['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)

#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'
mediana_valorTotal = dados['valorTotalHomologado'].median()

# Imputar os valores ausentes com a mediana
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)


# Colunas a serem excluídas
colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modalidadeId',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]

# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.

# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

colunas_a_excluir = [
    ['tipoInstrumentoConvocatorioNome',
        'situacaoCompraId',
        'orgaoEntidade.cnpj',
       'orgaoEntidade.poderId', 
       'orgaoEntidade.esferaId', 
       'amparoLegal.codigo', 
       'unidadeOrgao.ufNome',
     'unidadeOrgao.codigoUnidade',
     'unidadeOrgao.codigoIbge']
]

#As colunas acima são reduntantes, pois todas apresentam uma coluna correspondente no formato int que representa o id da informação.
# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

#filtrando as variáveis com todos os valores nulos
colunas_nulas = dados.columns[dados.isnull().all()]

dados = dados.drop(columns=colunas_nulas)

# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')


#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

#Preenchendo informações faltantes. 
dados['informacaoComplementar'].fillna("Sem informação", inplace=True)
dados['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)

#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'
mediana_valorTotal = dados['valorTotalHomologado'].median()

# Imputar os valores ausentes com a mediana
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)


# Colunas a serem excluídas
colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modalidadeId',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]

# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.

# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

colunas_a_excluir = [
    'tipoInstrumentoConvocatorioNome',
        'situacaoCompraId',
        'orgaoEntidade.cnpj',
       'orgaoEntidade.poderId', 
       'orgaoEntidade.esferaId', 
       'amparoLegal.codigo', 
       'unidadeOrgao.ufNome',
     'unidadeOrgao.codigoUnidade',
     'unidadeOrgao.codigoIbge'
]

#As colunas acima são reduntantes, pois todas apresentam uma coluna correspondente no formato int que representa o id da informação.
# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

# %%
df_reduzido.describe

# %%
df_reduzido.describe()

# %%
categoricos = df_reduzido.select_dtypes(object)

# %%
categoricos

# %%
categoricos = df_reduzido.select_dtypes(include='object')

# %%
categoricos

# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

#filtrando as variáveis com todos os valores nulos
colunas_nulas = dados.columns[dados.isnull().all()]

dados = dados.drop(columns=colunas_nulas)

# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')


#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

#Preenchendo informações faltantes. 
dados['informacaoComplementar'].fillna("Sem informação", inplace=True)
dados['linkSistemaOrigem'].fillna("Link não disponível", inplace=True)

#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'
mediana_valorTotal = dados['valorTotalHomologado'].median()

# Imputar os valores ausentes com a mediana
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)


# Colunas a serem excluídas
colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modalidadeId',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]

# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.

# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

colunas_a_excluir = [
    'tipoInstrumentoConvocatorioNome',
        'situacaoCompraId',
        'orgaoEntidade.cnpj',
       'orgaoEntidade.poderId', 
       'orgaoEntidade.esferaId', 
       'amparoLegal.codigo', 
       'unidadeOrgao.ufNome',
     'unidadeOrgao.codigoUnidade',
     'unidadeOrgao.codigoIbge'
]

#As colunas acima são reduntantes, pois todas apresentam uma coluna correspondente no formato int que representa o id da informação.
#Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

# %%
categoricos = df_reduzido.select_dtypes(include='object')


# %%
categoricos

# %%
df_reduzido

# %%
import pandas as pd 
dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')
#filtrando as variáveis com todos os valores nulos

colunas_nulas = dados.columns[dados.isnull().all()]
dados = dados.drop(columns=colunas_nulas)
# Lista das colunas a serem convertidas para datetime

colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']
# Converte as colunas especificadas para o tipo datetime

for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')
#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover

colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]
# Removendo as colunas do DataFrame

dados = dados.drop(columns=colunas_a_remover)
#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'

mediana_valorTotal = dados['valorTotalHomologado'].median()
# Imputar os valores ausentes com a mediana

dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)
# Colunas a serem excluídas

colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modalidadeId',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]
# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.
# Excluir colunas do DataFrame

df_reduzido = dados.drop(columns=colunas_a_excluir)
colunas_a_excluir = [
    'tipoInstrumentoConvocatorioNome',
        'situacaoCompraId',
        'orgaoEntidade.cnpj',
       'orgaoEntidade.poderId', 
       'orgaoEntidade.esferaId', 
       'amparoLegal.codigo', 
       'unidadeOrgao.ufNome',
     'unidadeOrgao.codigoUnidade',
     'unidadeOrgao.codigoIbge'
]
#As colunas acima são reduntantes, pois todas apresentam uma coluna correspondente no formato int que representa o id da informação.
#Excluir colunas do DataFrame

df_reduzido = dados.drop(columns=colunas_a_excluir)

# %%
df_reduzido

# %%
import pandas as pd 
dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')
#filtrando as variáveis com todos os valores nulos

colunas_nulas = dados.columns[dados.isnull().all()]
dados = dados.drop(columns=colunas_nulas)
# Lista das colunas a serem convertidas para datetime

colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']
# Converte as colunas especificadas para o tipo datetime

for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')
#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover

colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]
# Removendo as colunas do DataFrame

dados = dados.drop(columns=colunas_a_remover)
#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'

mediana_valorTotal = dados['valorTotalHomologado'].median()
# Imputar os valores ausentes com a mediana

dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)
# Colunas a serem excluídas

colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modalidadeId',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]
# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.
# Excluir colunas do DataFrame

df_reduzido = dados.drop(columns=colunas_a_excluir)
colunas_redundantes_a_excluir = [
    'tipoInstrumentoConvocatorioNome',
        'situacaoCompraId',
        'orgaoEntidade.cnpj',
       'orgaoEntidade.poderId', 
       'orgaoEntidade.esferaId', 
       'amparoLegal.codigo', 
       'unidadeOrgao.ufNome',
     'unidadeOrgao.codigoUnidade',
     'unidadeOrgao.codigoIbge'
]
#As colunas acima são reduntantes, pois todas apresentam uma coluna correspondente no formato int que representa o id da informação.
#Excluir colunas do DataFrame

df_reduzido = dados.drop(columns=colunas_redundantes_a_excluir)

# %%
df_reduzido

# %%
df_reduzido.columns

# %%
import pandas as pd 

dados = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')

#filtrando as variáveis com todos os valores nulos
colunas_nulas = dados.columns[dados.isnull().all()]

dados = dados.drop(columns=colunas_nulas)

# Lista das colunas a serem convertidas para datetime
colunas_data = ['dataAberturaProposta','dataEncerramentoProposta', 'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao']

# Converte as colunas especificadas para o tipo datetime
for coluna in colunas_data:
    dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')


#excluindo colunas que apresentam 2966 valores perdidos,já que a base de dados apresenta 2967 valores
# Listando as colunas que você deseja remover
colunas_a_remover = [
    'orgaoSubRogado.razaoSocial', 
    'orgaoSubRogado.poderId', 
    'orgaoSubRogado.esferaId', 
    'unidadeSubRogada.ufNome', 
    'unidadeSubRogada.nomeUnidade', 
    'unidadeSubRogada.ufSigla', 
    'unidadeSubRogada.municipioNome',
    'orgaoSubRogado.cnpj', 
    'unidadeSubRogada.codigoUnidade',
    'unidadeSubRogada.codigoIbge'
]

# Removendo as colunas do DataFrame
dados = dados.drop(columns=colunas_a_remover)

#Considerando o grande número de outliers (389) optamos por preencher os dados a partir da mediana# Calcular a mediana da coluna 'valorTotalHomologado'
mediana_valorTotal = dados['valorTotalHomologado'].median()

# Imputar os valores ausentes com a mediana
dados['valorTotalHomologado'].fillna(mediana_valorTotal, inplace=True)

# Colunas a serem excluídas
colunas_a_excluir = [
    'informacaoComplementar',
    'linkSistemaOrigem',
    'numeroCompra',
    'numeroControlePNCP',
    'modalidadeNome',
    'modalidadeId',
    'modoDisputaNome',
    'usuarioNome',
    'amparoLegal.nome',
    'unidadeOrgao.ufSigla',
]

# As colunas a serem excluídas foram selecionadas com base na sua relevância e na possibilidade de redundância:
# 
# - 'informacaoComplementar': Pode não fornecer dados críticos para a análise de processos de dispensa.
# - 'linkSistemaOrigem': Não é necessário para a análise em si e não agrega valor à compreensão dos dados.
# - 'numeroCompra': Redundante, pois o 'processo' já identifica unicamente cada compra.
# - 'numeroControlePNCP': Não é essencial para a análise e não agrega valor significativo.
# - 'modalidadeNome' e 'modoDisputaNome': Essas informações podem ser menos relevantes, visto que o foco é apenas nas dispensas.
# - 'usuarioNome': A identidade do usuário não é relevante para a análise do processo em si.
# - 'amparoLegal.nome': Pode ser considerado redundante se a descrição já fornecer a informação necessária sobre a base legal.
# - 'unidadeOrgao.ufSigla': Redundante em relação a 'unidadeOrgao.ufNome', que já fornece a informação completa.
# - 'unidadeOrgao.codigoUnidade': Não é crítico para a análise e pode ser dispensado.
# As colunas que permanecem fornecem informações essenciais e relevantes para a análise dos processos de dispensa de licitação, garantindo que os dados sejam mais claros e focados.

# Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_a_excluir)

colunas_redundantes_a_excluir = [
    'tipoInstrumentoConvocatorioNome',
        'situacaoCompraId',
        'orgaoEntidade.cnpj',
       'orgaoEntidade.poderId', 
       'orgaoEntidade.esferaId', 
       'amparoLegal.codigo', 
       'unidadeOrgao.ufNome',
     'unidadeOrgao.codigoUnidade',
     'unidadeOrgao.codigoIbge'
]

#As colunas acima são reduntantes, pois todas apresentam uma coluna correspondente no formato int que representa o id da informação.
#Excluir colunas do DataFrame
df_reduzido = dados.drop(columns=colunas_redundantes_a_excluir)


