import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd

def calcular_acuracia(previsao_modelo: list[int], resultado_real: list[int]) -> float:
    '''Calcula a acuracia do modelo'''
    acertos = 0
    for i in range(len(previsao_modelo)):
        if previsao_modelo[i] == resultado_real[i]:
            acertos += 1
    acuraccy = acertos/len(previsao_modelo)
    return acuraccy

def plotar_resultados_por_feature(df: pd.DataFrame, features: list[str], coluna_resultado: str) -> None:
    for feature in features:
        x = df[feature]
        y = df[coluna_resultado]
        plt.figure(figsize = (40,40))
        plt.scatter(x,y)
        plt.xlabel(f"{feature}")
        plt.ylabel(f"{coluna_resultado}")
        plt.title(f"Relacao {feature} x {coluna_resultado}")
        plt.show()

df_treino = pd.DataFrame({
    "Idade": [
        21, 22, 24, 25, 27, 28, 29, 30, 32, 33,
        34, 35, 38, 39, 40, 42, 45, 47, 48, 50,
        52, 55, 56, 58, 60, 61, 63
    ],
    "Renda_Mensal": [
        1100, 1300, 1700, 2100, 2300, 2600, 2800, 3000, 3200, 3400,
        3600, 3800, 4000, 4200, 4500, 4800, 5100, 5400, 5600, 6000,
        6300, 6600, 7000, 7200, 7600, 8000, 8300
    ],
    "Tempo_Emprego": [
        0, 1, 2, 2, 3, 3, 4, 4, 5, 6,
        6, 7, 8, 9, 9, 10, 12, 14, 15, 17,
        18, 20, 22, 23, 25, 27, 30
    ],
    "Historico_Inadimplencia": [
        1, 1, 1, 1, 1, 0, 1, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0
    ],
    "Aprovado": [
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1
    ]
})


features = ["Idade","Renda_Mensal","Tempo_Emprego","Historico_Inadimplencia"]
target = ["Aprovado"]

df_teste = pd.DataFrame({
    "Idade": [22, 25, 27, 30, 33, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 28, 34, 41, 47, 53, 59],
    "Renda_Mensal": [1200, 1800, 2200, 2600, 3000, 3300, 3700, 4000, 4300, 4700, 5100, 5500, 5900, 6300, 6700, 7200,2400, 3200, 4100, 4800, 6000, 6900],
    "Tempo_Emprego": [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 15, 17, 18, 20,2, 6, 10, 13, 16, 19],
    "Historico_Inadimplencia": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,0, 1, 1, 1, 0, 0],
    "Aprovado": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
})

model = tree.DecisionTreeClassifier()

X = df_treino[features]
y = df_treino[target]
model.fit(X,y)

teste_df = pd.DataFrame(df_teste)
X_teste = teste_df[features]
y_teste = teste_df[target]

predict = model.predict(X_teste)
acuraccy = calcular_acuracia(predict, list(y_teste[target[0]]))
print(f"Acuraccy: {acuraccy}")

#Plots do dataset de treinos
plotar_resultados_por_feature(df_treino, features, target[0])