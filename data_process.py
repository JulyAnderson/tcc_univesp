import pandas as pd 

df1 = pd.read_csv('dados_coletados_PNCP_ate_pagina_74_normalize.csv')
df2 = pd.read_csv('dados_coletados_PNCP_ate_pagina_74.csv')

print(df1.columns)
print(df2.columns)