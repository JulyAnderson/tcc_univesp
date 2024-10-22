import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import List, Dict
import logging
from pathlib import Path


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataPreprocessor:
    def __init__(self, input_file: str, robust:bool = False):
        """
        Inicializa o preprocessador de dados.
        
        Args:
            input_file (str): Caminho para o arquivo CSV de entrada
        """
        self.input_file = Path(input_file) if input_file else None
        self.data = None
        self.scaler = StandardScaler() if not robust else RobustScaler()
        
    def load_data(self) -> None:
        """Carrega os dados do arquivo CSV."""
        try:
            self.data = pd.read_csv(self.input_file, index_col='Unnamed: 0')
            logging.info(f"Dados carregados com sucesso. Shape: {self.data.shape}")
        except Exception as e:
            logging.error(f"Erro ao carregar dados: {e}")
            raise
            
    def remove_duplicates(self) -> None:
        """Remove linhas duplicadas."""
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        removed_rows = initial_rows - len(self.data)
        logging.info(f"Removidas {removed_rows} linhas duplicadas")
        
    def remove_null_columns(self) -> None:
        """Remove colunas com todos os valores nulos ou apenas um valor não nulo."""
        null_cols = self.data.columns[self.data.isnull().all()].tolist()
        single_value_cols = self.data.columns[self.data.notnull().sum() == 1].tolist()
        cols_to_remove = null_cols + single_value_cols
        
        self.data = self.data.drop(columns=cols_to_remove)
        logging.info(f"Removidas {len(cols_to_remove)} colunas com valores nulos/únicos")
        
    def remove_redundant_columns(self, redundant_cols: List[str]) -> None:
        """
        Remove colunas redundantes especificadas.
        
        Args:
            redundant_cols (List[str]): Lista de colunas a serem removidas
        """
        self.data = self.data.drop(columns=redundant_cols)
        logging.info(f"Removidas {len(redundant_cols)} colunas redundantes")
        
    def process_dates(self) -> None:
        """Processa e transforma colunas de data."""
        date_columns = [
            'dataAberturaProposta', 'dataEncerramentoProposta',
            'dataInclusao', 'dataPublicacaoPncp', 'dataAtualizacao'
        ]
        
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
        
        # Removendo colunas de data redundantes após análise de correlação
        self.data = self.data.drop(columns=['dataEncerramentoProposta'])
        logging.info("Processamento de datas concluído")
        
    def handle_missing_values(self) -> None:
        """Trata valores faltantes usando a mediana e remove outliers."""
        # Tratamento para valorTotalHomologado
        q1 = self.data['valorTotalHomologado'].quantile(0.25)
        q3 = self.data['valorTotalHomologado'].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_mask = (self.data['valorTotalHomologado'] < lower_bound) | \
                       (self.data['valorTotalHomologado'] > upper_bound)
        
        n_outliers = outliers_mask.sum()
        logging.info(f"Identificados {n_outliers} outliers em valorTotalHomologado")
        
        median_value = self.data['valorTotalHomologado'].median()
        self.data['valorTotalHomologado'].fillna(median_value, inplace=True)
        
    def encode_categorical_variables(self, encoding_cols: Dict[str, List]) -> None:
        """
        Realiza one-hot encoding para variáveis categóricas.
        
        Args:
            encoding_cols (Dict[str, List]): Dicionário com colunas e seus valores possíveis
        """
        for col, possible_values in encoding_cols.items():
            dummies = pd.get_dummies(self.data[col], prefix=col)
            
            # Garantir que todas as categorias esperadas existam
            for value in possible_values:
                col_name = f"{col}_{value}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
                    
            self.data = pd.concat([self.data, dummies], axis=1)
            self.data = self.data.drop(columns=[col])
            
        logging.info("Encoding de variáveis categóricas concluído")
        
    def normalize_numerical_features(self) -> None:
        """Normaliza features numéricas usando StandardScaler."""
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numerical_cols] = self.scaler.fit_transform(self.data[numerical_cols])
        logging.info("Normalização de features numéricas concluída")
        
    def save_processed_data(self, output_file: str) -> None:
        """
        Salva os dados processados em um arquivo CSV.
        
        Args:
            output_file (str): Caminho para o arquivo de saída
        """
        self.data.to_pickle(output_file)
        logging.info(f"Dados processados salvos em {output_file}")

    def return_df(self) -> pd.DataFrame:
        """Retorna o DataFrame processado."""
        return self.data

def preprocess_data(input_file: str, output_file: str, redundant_cols: List[str], encoding_config: Dict[str, List]):
    """
    Função para centralizar o processo de carregamento, limpeza e salvamento de dados.
    
    Args:
        input_file (str): Caminho do arquivo de entrada CSV.
        output_file (str): Caminho do arquivo CSV de saída.
        redundant_cols (List[str]): Lista de colunas redundantes a serem removidas.
        encoding_config (Dict[str, List]): Configuração para o encoding de variáveis categóricas.
    """
    try:
        # Instanciar e executar o preprocessador
        preprocessor = DataPreprocessor(input_file)
        preprocessor.load_data()
        preprocessor.remove_duplicates()
        preprocessor.remove_null_columns()
        preprocessor.process_dates()
        preprocessor.remove_redundant_columns(redundant_cols)
        preprocessor.handle_missing_values()
        preprocessor.encode_categorical_variables(encoding_config)
        preprocessor.normalize_numerical_features()
        preprocessor.save_processed_data(output_file)
        
        logging.info("Preprocessamento concluído com sucesso!")
        
        return preprocessor.return_df()
        
    except Exception as e:
        logging.error(f"Erro durante o preprocessamento: {e}")
        raise

def main():
    # Defina os parâmetros necessários
    input_file = 'dados/brutos/dados_coletados_PNCP_ate_pagina_74_normalize.csv'
    output_file = 'dados/processados/dados_processados.pkl'

     # Colunas redundantes a serem removidas
    redundant_cols = [
        'modalidadeId', 'modalidadeNome', 'situacaoCompraNome', 'usuarioNome',
        'orgaoEntidade.razaoSocial', 'informacaoComplementar', 'objetoCompra',
        'linkSistemaOrigem', 'unidadeOrgao.ufNome', 'unidadeOrgao.ufSigla',
        'amparoLegal.descricao', 'amparoLegal.nome', 'unidadeOrgao.municipioNome',
        'unidadeOrgao.nomeUnidade', 'tipoInstrumentoConvocatorioNome', 'modoDisputaNome',
        'orgaoEntidade.cnpj', 'numeroCompra', 'numeroControlePNCP', "processo",
        "unidadeOrgao.codigoUnidade"
    ]
    
    # Configuração para encoding categórico
    encoding_config = {
        'modoDisputaId': [4, 5],
        'situacaoCompraId': [1, 2, 3],
        'tipoInstrumentoConvocatorioCodigo': [2, 3],
        'amparoLegal.codigo': [18, 19, 20, 21, 22, 24, 36, 37, 38, 39, 41, 45],
        'orgaoEntidade.poderId': ['E', 'N', 'L', 'J'],
        'orgaoEntidade.esferaId': ['F', 'M', 'E', 'N', 'D']
    }

    # Chame a função de preprocessamento
    try:
        preprocessed_df = preprocess_data(input_file, output_file, redundant_cols, encoding_config)

        # Exiba as primeiras linhas do DataFrame processado
        print(preprocessed_df.head())
        
    except Exception as e:
        logging.error(f"Erro ao executar o preprocessamento: {e}")

if __name__ == "__main__":
    main()



#A coluna 'modalidadeId' possui o valor fixo 8, indicando que todas as linhas correspondem à modalidade de dispensa de licitações, portanto, não é relevante para a análise.
#As colunas 'modalidadeNome', 'situacaoCompraNome', 'usuarioNome', e 'orgaoEntidade.razaoSocial' são redundantes, pois já possuímos o 'modalidadeId' e 'situacaoCompraId' com
#a mesma informação em formato numérico.
#Outras colunas, como 'informacaoComplementar', 'objetoCompra', 'linkSistemaOrigem', e detalhes sobre localização como 'unidadeOrgao.ufNome', 'unidadeOrgao.ufSigla', entre outras,
#não são necessárias para a análise pretendida.
