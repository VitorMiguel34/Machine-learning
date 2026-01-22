import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import numpy as np

# ============================================================
# EXERCÍCIO: REGRESSÃO LINEAR — PREVISÃO DE SALÁRIO
# ============================================================
#
# ENUNCIADO:
# Você trabalha em uma plataforma de cursos online e precisa
# desenvolver um modelo de Machine Learning capaz de prever
# o salário mensal de uma pessoa com base em seu perfil
# profissional.
#
# O salário será estimado a partir das seguintes variáveis:
# - Anos de estudo
# - Anos de experiência profissional
# - Horas trabalhadas por semana
#
# ------------------------------------------------------------
# OBJETIVO:
# Treinar e avaliar um modelo de Regressão Linear utilizando
# analisando tanto métricas de desempenho quanto a 
# interpretação dos coeficientes.

df = pd.DataFrame({
    "Anos_Estudo": [8, 10, 12, 14, 16, 18, 11, 13, 15, 17, 9, 7, 19, 20, 6],
    "Anos_Experiencia": [1, 2, 3, 5, 6, 8, 4, 5, 7, 9, 2, 1, 10, 12, 0],
    "Horas_Semanais": [30, 35, 40, 40, 45, 50, 38, 42, 45, 50, 32, 28, 55, 60, 25],
    "Salario": [1800, 2200, 2800, 3500, 4200, 5200, 3000, 3400, 4600, 5800, 2100, 1600, 6500, 7200, 1400]
})

features = ["Anos_Estudo","Anos_Experiencia","Horas_Semanais"]
target = ["Salario"]

for feature in features:
    plt.figure(figsize=(12,6))
    plt.scatter(df[feature],df["Salario"])
    plt.xlabel(f"{feature}")
    plt.ylabel("Salario")
    plt.title(f"Relacao {feature} x {target[0]}")
    plt.show()

tamanho_treino = (int(len(df)*0.7))
tamanho_teste = len(df) - tamanho_treino
df_treino, df_teste = train_test_split(df, train_size=0.3, random_state=42)
X_treino = df_treino[features]
y_treino = df_treino[target]

model = linear_model.LinearRegression()
model.fit(X_treino,y_treino)

X_teste = df_teste[features]
y_teste = df_teste[target]
previsoes = model.predict(X_teste)

mse = mean_squared_error(y_true=y_teste[target[0]],y_pred=previsoes)
print(f"Mean Squared Error: {mse}")
sse = mse * tamanho_teste
print(f"Sum of Squared Errors: {sse}")

previsao_treino = model.predict(X_treino)
print("Previsao do treino:")
print(previsao_treino)
print("Valores reais:")
print(y_treino)