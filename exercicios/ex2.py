from sklearn import linear_model
import pandas as pd
import numpy as np

#Treinando um modeo com duas features

df = pd.DataFrame({
    "tamanho": [50,60,70,80],
    "quartos": [1,2,2,3],
    "preco": [150,180,200,250]
})

tamanhos = np.array(df["tamanho"])
quartos = np.array(df["quartos"])

X = np.array([[tamanhos[i],quartos[i]] for i in range(len(tamanhos))])
y = np.array(df["preco"])

model = linear_model.LinearRegression()
model.fit(X,y)

print(f"Coeficientes: {model.coef_}")
print(f"Interceptor: {model.intercept_}")

previsoes = np.array(model.predict(X))

#Calculando a media da soma dos erros quadráticos

vetor_diferenca = y - previsoes

#Deixando todos os valores vetor diferenca positivos
vetor_diferenca = abs(vetor_diferenca)

mse = sum(vetor_diferenca)/(len(previsoes))
print(f"MSE = {mse}")