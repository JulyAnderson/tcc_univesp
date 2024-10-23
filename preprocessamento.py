import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

dados = pd.read_csv("dados/brutos/dados_colunas_normalizadas.csv", index_col="Unnamed: 0")

# remover duplicadas
dados = dados.drop_duplicates()

#  0   valorTotalHomologado               2543 non-null   float64
#  21  valorTotalEstimado                 2967 non-null   float64
# Estatísticas descritivas da coluna valorTotalHomologado
dados['valorTotalHomologado'].describe()

# Criar um histograma da coluna valorTotalHomologado
dados['valorTotalHomologado'].hist(bins=10)  # bins define o número de "caixas" no gráfico
plt.title('Distribuição de valorTotalHomologado')
plt.xlabel('Valor Total Homologado')
plt.ylabel('Frequência')
plt.savefig("images/hist_distribuicao_valotHomologado.png")
plt.show()

# Criar um boxplot
dados.boxplot(column='valorTotalHomologado')
plt.title('Boxplot de valorTotalHomologado')
plt.savefig("images/box_plot_valorHomologado.png")
plt.show()
#
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
plt.savefig("images/violin_valorHomologado.png")
plt.show()

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
plt.show("images/hist_plot_valorHomologado_sns.png")
plt.show()

#considerando a alta correlação e observando que valor estimado e valor homolado são iguais em todas ocorrencias
#exceto quando o valorTotalHomolagado está ausente, optaremos pela remoção do valor totalHomologado, que possui 326 dados ausentes

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
plt.savefig("images/duracoes.png")
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
        plt.savefig("images/correlacoes.png")
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
        plt.savefig("images/top_correlacoes.png")
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

# considerando as correlações, os seguintes dados serão suprimidos por apresentarem o mesmo comportamento sempre. 
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

dados.to_pickle('dados/processados/Eda.pkl')
