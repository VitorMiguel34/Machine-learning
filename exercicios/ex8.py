import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

def calcular_soma_dos_erros_quadraticos(previsao: list[int|float], valores_reais: list[int|float]):
    soma = 0
    for i,valor in enumerate(previsao):
        soma += (valor[0] - valores_reais[i])**2
    return soma

df = pd.read_csv("dados/ex8.csv")
features = list(df.columns)
features.remove("Preco")
target = ["Preco"]

tamanho_df = len(df)
quantidade_teste = int(0.3*tamanho_df)

X_treino = df[features].iloc[quantidade_teste:]
y_treino = df[target].iloc[quantidade_teste:]

X_teste = df[features].iloc[:quantidade_teste]
y_teste = df[target].iloc[:quantidade_teste]

model = linear_model.LinearRegression()
model.fit(X_treino,y_treino)

previsao_treino = model.predict(X_treino)
previsao_teste = model.predict(X_teste)

erro_treino = calcular_soma_dos_erros_quadraticos(previsao_treino, list(y_treino["Preco"]))
erro_teste = calcular_soma_dos_erros_quadraticos(previsao_teste, list(y_teste["Preco"]))

print(f"Soma dos erros quadraticos no treino: {erro_treino:.5f}")
print(f"Soma das erros quadraticos no teste: {erro_teste:.5f}")

for feature in features:
    plt.figure(figsize=(12,6))
    plt.scatter(df[feature],df["Preco"])
    plt.xlabel(f"{feature}")
    plt.ylabel("Preço")
    plt.title(f"{feature} x Preço")
    plt.show()