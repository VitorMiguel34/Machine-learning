import pandas as pd

df = pd.read_csv("../dados/points.csv")

#Tabela de frequencia absoluta
freq_produto = df.groupby("descProduto")[["idTransacao"]].count()

#Reorganizando df
freq_produto.rename(columns={"idTransacao":"freq_abs"}, inplace=True)
freq_produto.sort_values(by="freq_abs", inplace=True, ascending=False)

#Coluna de frequência absoluta acumulada:
freq_produto["freq_abs_cum"] = freq_produto["freq_abs"].cumsum()

#Coluna de frenquência relativa
freq_produto["freq_rel"] = freq_produto["freq_abs"]/freq_produto["freq_abs"].sum()

#Coluna de frenquencia relativa acumulada
freq_produto["freq_rel_cum"] = freq_produto["freq_rel"].cumsum()

print(freq_produto)

freq_categoria = df.groupby("descCategoriaProduto")[["idTransacao"]].count()
freq_categoria.rename(columns={"idTransacao":"freq_abs"},inplace=True)
freq_categoria.sort_values(by="freq_abs", inplace=True, ascending=False)

freq_categoria["freq_abs_cum"] = freq_categoria["freq_abs"].cumsum()
freq_categoria["freq_rel"] = freq_categoria["freq_abs"]/freq_categoria["freq_abs"].sum()
freq_categoria["freq_rel_cum"] = freq_categoria["freq_rel"].cumsum()

print(freq_categoria)
