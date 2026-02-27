import pandas as pd
from k_means import distancia_vetorial
from sklearn.model_selection import train_test_split
import os

class KNN:
    def __init__(self, k: int):
        self.k = k
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def predict(self,X_teste: pd.DataFrame):
        resultados = list()
        X = X_teste.copy()

        for c in range(len(X)):
            x = X.iloc[c]
            distancias = list()
            for i in range(len(self.X)):
                distancias.append({
                    "id":i,
                    "distancia":distancia_vetorial(x,self.X.iloc[i])
                })
            distancias = sorted(distancias, key=lambda x: x["distancia"])
            k_vizinhos = distancias[:self.k]

            classes = dict()
            for vizinho in k_vizinhos:
                classe  = self.y.iloc[vizinho["id"]]
                if classe in classes:
                    classes[classe] += 1
                else:
                    classes[classe] = 1
            
            resultados.append(max(classes, key=classes.get))
        
        return resultados
        
        
df = pd.DataFrame({
    "idade": [
        18, 20, 22, 19, 23, 21,
        30, 32, 35, 33, 31, 34,
        45, 48, 50, 47, 52, 49
    ],
    "renda_mensal": [
        1200, 1500, 1300, 1100, 1600, 1400,
        4000, 4500, 4200, 4800, 4600, 4300,
        10000, 12000, 15000, 11000, 14000, 13000
    ],
    "gasto_mensal": [
        300, 400, 350, 250, 420, 380,
        1500, 1800, 1700, 1600, 1750, 1650,
        4000, 5000, 5500, 4800, 5300, 5100
    ],
    "score_credito": [
        400, 450, 420, 390, 460, 430,
        650, 700, 720, 680, 690, 710,
        800, 820, 850, 830, 870, 860
    ],
    "classe": [
        "Basico","Basico","Basico","Basico","Basico","Basico",
        "Intermediario","Intermediario","Intermediario","Intermediario","Intermediario","Intermediario",
        "Premium","Premium","Premium","Premium","Premium","Premium"
    ]
})

target = "classe"
features = list(df.columns)
features.remove(target)
df_treino,df_teste = train_test_split(df,test_size=0.3)
X_treino = df_treino[features]
y_treino = df_treino[target]
X_teste = df_teste[features]
y_teste = df_teste[target]

model = KNN(3)
model.fit(X_treino,y_treino)

predict = model.predict(X_teste)
df_comparacao = pd.DataFrame({
    "Previsão":predict,
    "Real":list(y_teste)
})

os.system("clear")
print(df_comparacao)