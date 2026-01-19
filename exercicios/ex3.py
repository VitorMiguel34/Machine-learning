from sklearn import linear_model
import pandas as pd
import numpy as np

#Fazendo classificaçāo binária simples com modelo de regressāo logística 

df = pd.DataFrame({
    "horas_estudo": [1,2,3,4,5,6],
    "aprovado": [0,0,0,1,1,1]
})

X = df[["horas_estudo"]]
y = df["aprovado"]

model = linear_model.LogisticRegression(random_state=1)
model.fit(X,y)

prever = [[2.6],[6.4],[7.8]]
previsao_probabilidade = model.predict_proba(prever)

limiar = 0.7

for index,value in enumerate(previsao_probabilidade):
    print(f"Probabilidades de {prever[index][0]}:")
    print(f"Reprovado: {value[0]}")
    print(f"Aprovado: {value[1]}")

    resultado = "Aprovado pelo limiar" if value[1] >= limiar else "Reprovado pelo limiar"
    print(f"Resultado: {resultado}")
    print("...")


