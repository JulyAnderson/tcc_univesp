import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

BASE_URL = "https://pncp.gov.br/api/consulta/v1/"

# Define os parametros da api
params = {
    "dataInicial": "20230101",  
    "dataFinal": "20231231",    
    "codigoModalidadeContratacao": 8,  # 8 corresponds to Dispensa de Licitação
    "pagina": 75
}

# Iniciando a lista para armazenar os dados
todos_os_dados = []

# Função para consumir a API
def fetch_data(params, retries=3):
    for attempt in range(retries):
        try:
            print(f"Requesting page {params['pagina']} with parameters: {params}")
            response = requests.get(BASE_URL + "contratacoes/publicacao", params=params)
            
            if response.status_code == 200:
                return response.json()  # Retorna JSON se sucesso
            else:
                print(f"Error: {response.status_code}. Response: {response.text}. Attempt {attempt+1}/{retries}")
        except requests.RequestException as e:
            print(f"Request error: {e}. Attempt {attempt+1}/{retries}")
        time.sleep(1)  # Aguarda antes de tentar novamente
    return None  # Retorna None se todas as tentivas falharem

# Function to save partial results to a file
def save_partial_results(data, filename="partial_results.json"):
    with open(filename, "a") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")

# Primeira chamada da API
result = fetch_data(params)

if result:
    todos_os_dados.extend(result.get('data', []))  # Coletando a primeira página dos dados
    total_paginas = result.get('totalPaginas', 1)  # Obtendo o total de paginas
    
    # Salva a primeira págia de dados
    save_partial_results(result.get('data', []))

    # Loop enquanto existir páginas
    for pagina in range(2, total_paginas + 1):
        params["pagina"] = pagina  # Atualiza o número de página
        
        result = fetch_data(params)
        if result:
            todos_os_dados.extend(result.get('data', []))
            
            # Salva cada página de forma incremental.
            save_partial_results(result.get('data', []))
        else:
            print(f"Failed to retrieve data for page {pagina}")

    print(f"Total records collected: {len(todos_os_dados)}")
else:
    print("Failed to retrieve any data.")


import pandas as pd
import json
import re

# Lista para armazenar os dados
dados = []

# Carregue o arquivo JSON
with open('partial_results.json', 'r', encoding='Windows-1252') as f:
    # Lê todo o conteúdo
    conteudo = f.read()
    # Substituir vírgulas duplicadas usando regex
    conteudo = re.sub(r'},\s*,', '},', conteudo)
    # Divide por colchetes (assumindo que não há colchetes aninhados)
    objetos = conteudo.split('][')
    for objeto in objetos:
        # Corrige a formatação para cada objeto
        objeto = objeto.strip().strip('[]')
        if objeto:  # Se não estiver vazio
            try:
                dados.extend(json.loads(f"[{objeto}]"))  # Adiciona colchetes para formar uma lista
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar JSON: {e}")

# Converta a lista de objetos em um DataFrame
df = pd.DataFrame(dados)

# Exiba o DataFrame
df

df.to_csv("dados_coletados_PNCP_ate_pagina_74.csv")

df_normalize = pd.json_normalize(dados)
df.to_csv("dados_coletados_PNCP_ate_pagina_74_normalize.csv")

df_normalize

