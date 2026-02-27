from typing import Iterable
import pandas as pd

def produto_escalar(v1: Iterable[float],v2: Iterable[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Os vetores não estão no mesmo espaço, portanto não é possível calcular o produto interno entre eles")
    resultado = 0
    for i in range(len(v1)):
        resultado += v1[i]*v2[i]
    return resultado

def distancia_vetorial(v1: Iterable[float], v2: Iterable[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Os vetores não estão no mesmo espaço, portanto não é possível calcular a distância entre eles")
    vetor_diferenca = [v1[i] - v2[i] for i in range(len(v1))]
    distancia = produto_escalar(vetor_diferenca,vetor_diferenca)**(1/2)
    return distancia

class KMeans:
    def __init__(self, k: int, max_iter: int = 100):
        self.k = k
        self.results = None
        self.centroides = [None for i in range(self.k)]
        self.max_iter = max_iter

    def fit(self,df: pd.DataFrame) -> None:
        dfx = df.copy()
        self.centroides = list(df.sample(self.k).values)

        for _ in range(self.max_iter):
            resultados = []

            for i in range(len(df)):
                distancia_minima = distancia = centroide_mais_proximo = 0
                v = list(df.iloc[i])

                for num_centroide,centroide in enumerate(self.centroides):
                    distancia = distancia_vetorial(v,centroide)
                    if num_centroide == 0:
                        distancia_minima = distancia
                        continue
                    if distancia < distancia_minima:
                        centroide_mais_proximo = num_centroide
                        distancia_minima = distancia

                resultados.append(centroide_mais_proximo)
            
            dfx["grupo"] = resultados

            centroides_antigos = self.centroides

            for i in range(self.k):
                if dfx[dfx["grupo"] == i].empty:
                    continue
                self.centroides[i] = list(dfx[dfx["grupo"] == i].drop(columns=["grupo"]).mean()) 

            if centroides_antigos == self.centroides:
                break          

        resultados_otimizados = resultados
        return resultados_otimizados


df = pd.DataFrame({
    "idade": [
        18, 20, 19, 22, 21, 23,
        30, 32, 35, 33, 31, 34,
        40, 42, 45, 43, 41, 44
    ],
    
    "peso_kg": [
        60, 65, 58, 62, 64, 59,
        75, 78, 80, 77, 76, 79,
        95, 100, 98, 102, 97, 105
    ],
    
    "altura_cm": [
        168, 170, 165, 172, 169, 167,
        175, 178, 180, 176, 174, 179,
        182, 185, 183, 187, 184, 188
    ],
    
    "frequencia_treino_semana": [
        2, 3, 2, 3, 2, 3,
        4, 5, 5, 4, 5, 4,
        6, 6, 7, 6, 7, 6
    ],
    
    "gasto_suplementos_mes": [
        50, 80, 40, 70, 60, 55,
        200, 250, 300, 220, 270, 260,
        600, 750, 700, 800, 720, 850
    ]
})

num_grupos = 3
model = KMeans(k=num_grupos)
grupos = model.fit(df)
df["grupo"] = grupos

for i in range(num_grupos):
    print(f"Grupo {i}")
    if not df[df["grupo"] == i].empty:
        print(df[df["grupo"] == i])
    else:
        print("Esse grupo nao possui individuos ligadas a ele")
    print()

            

            


    



