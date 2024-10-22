# models/autoencoder.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class AnomalyAutoencoder:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            # Encoder
            Dense(64, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            
            # Decoder
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(self.input_dim, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def fit(self, X, **kwargs):
        return self.model.fit(X, X, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)

# preprocessing/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = [
            'modalidadeNome', 
            'modoDisputaNome', 
            'tipoInstrumentoConvocatorioNome'
        ]
        
    def _process_monetary_values(self, df):
        df = df.copy()
        monetary_columns = ['valorTotalHomologado', 'valorTotalEstimado']
        
        for col in monetary_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['valorTotalEstimado'] = df['valorTotalEstimado'].replace(0, np.nan)
        return df
    
    def _create_features(self, df):
        df['valor_diff'] = abs(df['valorTotalHomologado'] - df['valorTotalEstimado'])
        df['valor_ratio'] = (df['valorTotalHomologado'] / df['valorTotalEstimado']).clip(-10, 10)
        return df
    
    def _encode_categories(self, df):  # This was previously named _encode_features
        for col in self.categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].fillna('UNKNOWN'))
        return df
    
    def fit_transform(self, df):
        df = self._process_monetary_values(df)
        df = self._create_features(df)
        df = self._encode_categories(df)  # Fixed method name here
        
        features = self._get_feature_columns(df)
        X = df[features].fillna(df[features].mean())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        return self.scaler.fit_transform(X), df
    
    def transform(self, df):
        df = self._process_monetary_values(df)
        df = self._create_features(df)
        df = self._encode_categories(df)  # Fixed method name here
        
        features = self._get_feature_columns(df)
        X = df[features].fillna(df[features].mean())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        return self.scaler.transform(X), df
    
    def _get_feature_columns(self, df):
        return [
            'valorTotalHomologado', 
            'valorTotalEstimado',
            'valor_diff',
            'valor_ratio',
            'modalidadeNome',
            'modoDisputaNome',
            'tipoInstrumentoConvocatorioNome'
        ]


# visualization/plotter.py
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyPlotter:
    @staticmethod
    def plot_anomaly_distribution(anomaly_scores):
        plt.figure(figsize=(10, 6))
        sns.histplot(anomaly_scores, bins=50)
        plt.title('Distribuição dos Scores de Anomalia')
        plt.xlabel('Score de Anomalia')
        plt.ylabel('Frequência')
        plt.show()
    
    @staticmethod
    def plot_feature_comparison(anomalias, normais, feature):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[anomalias[feature], normais[feature]], 
                   labels=['Anomalias', 'Normais'])
        plt.title(f'Comparação de {feature} entre Anomalias e Normais')
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df):
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlação para Anomalias')
        plt.show()

# analysis/evaluator.py
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np


class EnhancedReportGenerator:
    def __init__(self, df, anomaly_scores, is_anomaly):
        self.df = df
        self.anomaly_scores = anomaly_scores
        self.is_anomaly = is_anomaly
        self.anomalias = df[is_anomaly]
        self.normais = df[~is_anomaly]
        self.features_numericas = self._get_numeric_features()
    
    def _get_numeric_features(self):
        return self.df.select_dtypes(include=[np.number]).columns

    def _calculate_statistics(self):
        """
        Calcula estatísticas básicas dos scores de anomalia
        """
        return {
            'média': float(np.mean(self.anomaly_scores)),
            'mediana': float(np.median(self.anomaly_scores)),
            'desvio_padrão': float(np.std(self.anomaly_scores)),
            'percentil_95': float(np.percentile(self.anomaly_scores, 95)),
            'percentil_99': float(np.percentile(self.anomaly_scores, 99)),
            'max': float(np.max(self.anomaly_scores)),
            'min': float(np.min(self.anomaly_scores)),
            'total_registros': len(self.df),
            'total_anomalias': len(self.anomalias)
        }
    
    def _run_statistical_tests(self):
        """
        Executa testes estatísticos para cada feature numérica
        """
        results = {}
        for feature in self.features_numericas:
            try:
                stat, pvalue = stats.mannwhitneyu(
                    self.anomalias[feature].dropna(),
                    self.normais[feature].dropna()
                )
                results[feature] = {
                    'estatistica': float(stat),
                    'p_value': float(pvalue)
                }
            except:
                continue
        return results

    def _calculate_silhouette(self):
        """
        Calcula o Silhouette Score e métricas relacionadas
        """
        try:
            # Preparar dados para o cálculo do Silhouette Score
            X = self.df[self.features_numericas].fillna(0)
            X = StandardScaler().fit_transform(X)
            
            # Calcular Silhouette Score
            sil_score = silhouette_score(X, self.is_anomaly)
            
            # Calcular scores por cluster
            sil_samples = silhouette_samples(X, self.is_anomaly)
            
            return {
                'score': float(sil_score),
                'normal_avg': float(np.mean(sil_samples[~self.is_anomaly])),
                'anomaly_avg': float(np.mean(sil_samples[self.is_anomaly])),
                'normal_std': float(np.std(sil_samples[~self.is_anomaly])),
                'anomaly_std': float(np.std(sil_samples[self.is_anomaly]))
            }
        except Exception as e:
            print(f"Erro ao calcular Silhouette Score: {str(e)}")
            return {
                'score': None,
                'normal_avg': None,
                'anomaly_avg': None,
                'normal_std': None,
                'anomaly_std': None
            }

    def _interpret_silhouette(self, score):
        """
        Interpreta o valor do Silhouette Score
        """
        if score is None:
            return "(não foi possível calcular)"
        elif score > 0.7:
            return "(excelente separação)"
        elif score > 0.5:
            return "(boa separação)"
        elif score > 0.3:
            return "(separação moderada)"
        else:
            return "(separação fraca)"

    def _format_statistical_tests(self, tests):
        """
        Formata os resultados dos testes estatísticos
        """
        formatted = []
        for feature, result in tests.items():
            significance = self._interpret_p_value(result['p_value'])
            formatted.append(f"- **{feature}**: p-value = {result['p_value']:.4f} ({significance})")
        return "\n".join(formatted)

    def _interpret_p_value(self, p_value):
        """
        Interpreta o valor-p dos testes estatísticos
        """
        if p_value < 0.001:
            return "altamente significativo"
        elif p_value < 0.01:
            return "muito significativo"
        elif p_value < 0.05:
            return "significativo"
        else:
            return "não significativo"

    def _format_silhouette_analysis(self, silhouette):
        """
        Formata a análise do Silhouette Score
        """
        if silhouette['score'] is None:
            return "Não foi possível calcular o Silhouette Score para este conjunto de dados."
        
        return f"""#### Detalhamento do Silhouette Score
- **Score Global**: {silhouette['score']:.3f} {self._interpret_silhouette(silhouette['score'])}
- **Dados Normais**:
  - Média: {silhouette['normal_avg']:.3f}
  - Desvio Padrão: {silhouette['normal_std']:.3f}
- **Anomalias**:
  - Média: {silhouette['anomaly_avg']:.3f}
  - Desvio Padrão: {silhouette['anomaly_std']:.3f}

{self._generate_silhouette_recommendations(silhouette)}"""

    def _generate_silhouette_recommendations(self, silhouette):
        """
        Gera recomendações baseadas no Silhouette Score
        """
        if silhouette['score'] is None:
            return ""
        
        recommendations = ["#### Recomendações baseadas no Silhouette Score"]
        
        if silhouette['score'] < 0.3:
            recommendations.append("""
- Considerar ajustar os parâmetros do modelo de detecção
- Avaliar a inclusão de features adicionais
- Verificar se há subgrupos naturais nos dados que precisam ser tratados separadamente""")
        elif silhouette['score'] < 0.5:
            recommendations.append("""
- Investigar features que podem melhorar a separação
- Considerar técnicas de feature engineering
- Avaliar se o threshold de anomalia está adequado""")
        else:
            recommendations.append("""
- Manter os parâmetros atuais do modelo
- Documentar as features mais relevantes para a separação
- Implementar monitoramento contínuo deste score""")
        
        return "\n".join(recommendations)

    def _generate_insights(self, stats, tests, silhouette):
        """
        Gera insights baseados em todas as análises
        """
        insights = ["## Insights Principais\n"]
        
        # Análise da distribuição
        if stats['média'] > stats['mediana'] * 2:
            insights.append("- **Distribuição Assimétrica**: Forte presença de outliers positivos")
        
        # Análise da variabilidade
        if stats['desvio_padrão'] > stats['média'] * 2:
            insights.append("- **Alta Variabilidade**: Grande dispersão nos scores de anomalia")
        
        # Análise do Silhouette Score
        if silhouette['score'] is not None:
            if silhouette['score'] > 0.5:
                insights.append(f"- **Boa Separação**: Silhouette Score de {silhouette['score']:.3f} indica clara distinção entre anomalias e dados normais")
            elif silhouette['score'] < 0.3:
                insights.append(f"- **Separação Fraca**: Silhouette Score de {silhouette['score']:.3f} sugere necessidade de ajustes no modelo")
        
        # Análise da significância
        all_significant = all(test['p_value'] < 0.05 for test in tests.values())
        if all_significant:
            insights.append("- **Significância Global**: Todas as métricas apresentam diferenças estatisticamente significativas")
        
        return "\n".join(insights)

    def generate_markdown_report(self):
        """
        Gera o relatório completo em formato Markdown
        """
        stats = self._calculate_statistics()
        tests = self._run_statistical_tests()
        silhouette = self._calculate_silhouette()
        
        return f"""# Relatório de Detecção de Anomalias em Compras Públicas

## Resumo Executivo
- **Total de registros analisados**: {stats['total_registros']:,}
- **Anomalias detectadas**: {stats['total_anomalias']:,} ({stats['total_anomalias']/stats['total_registros']*100:.2f}% do total)
- **Qualidade da Separação (Silhouette)**: {silhouette['score']:.3f} {self._interpret_silhouette(silhouette['score'])}

## Análise Estatística dos Scores de Anomalia

### Métricas Principais
| Métrica | Valor | Interpretação |
|---------|--------|---------------|
| Média | {stats['média']:.4f} | Score médio de anomalia |
| Mediana | {stats['mediana']:.4f} | 50% dos casos têm score abaixo deste valor |
| Desvio Padrão | {stats['desvio_padrão']:.4f} | Medida de variabilidade nos scores |
| Percentil 95 | {stats['percentil_95']:.4f} | Limite para casos moderadamente anômalos |
| Percentil 99 | {stats['percentil_99']:.4f} | Limite para casos extremamente anômalos |

### Análise de Silhouette Score
{self._format_silhouette_analysis(silhouette)}

### Análise de Significância Estatística
{self._format_statistical_tests(tests)}

{self._generate_insights(stats, tests, silhouette)}"""

# main.py
def main():
    try:
        # Load data
        df = pd.read_csv('dados/brutos/dados_coletados_PNCP_ate_pagina_74_normalize.csv')
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_scaled, df_processed = preprocessor.fit_transform(df)
        
        # Train autoencoder
        autoencoder = AnomalyAutoencoder(X_scaled.shape[1])
        autoencoder.fit(
            X_scaled,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Detect anomalies
        reconstructed = autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        threshold_value = np.percentile(mse, 95)
        is_anomaly = mse > threshold_value
        
        # Evaluate results
        evaluator = EnhancedReportGenerator(df_processed, mse, is_anomaly)
        report = evaluator.generate_markdown_report()
        print(report)
        
        # Plot results
        plotter = AnomalyPlotter()
        plotter.plot_anomaly_distribution(mse)
        plotter.plot_correlation_matrix(df_processed[is_anomaly])
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()