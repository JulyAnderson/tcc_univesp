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
