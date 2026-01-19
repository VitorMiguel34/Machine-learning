import pandas as pd
from ex6 import arvore1,arvore2,arvore3

estuda = int(input("Estuda? (1-sim,0-nao)"))
trabalha = int(input("Trabalha? (1-sim,0-nao)"))
estado_civil = int(input("Casado? (1-sim,0-nao)"))

infos_usuario = pd.DataFrame(
    [[estuda, trabalha, estado_civil]],
    columns=["estuda", "trabalha", "estado_civil"]
)

#Classficacoes entre Jovem,Adulto e Idoso
previsao_arvore1 = arvore1.predict(infos_usuario)[0]
previsao_arvore2 = arvore2.predict(infos_usuario)[0]
previsao_arvore3 = arvore3.predict(infos_usuario)[0]

def mostrar_probabilidades(arvore,classes: list[str], infos_usuario: list[list[int]]) -> None:
    probs = arvore.predict_proba(infos_usuario)[0]
    for classe, prob in zip(classes,probs):
        print(f"{classe}: {prob}")

#Probabilidades
probs_arvore1 = arvore1.predict_proba(infos_usuario)
probs_arvore2 = arvore2.predict_proba(infos_usuario)
probs_arvore3 = arvore3.predict_proba(infos_usuario)
classes = arvore1.classes_

print("Previsoes:")
print(f"Previsao da arvore 1: {previsao_arvore1}")
print(f"Previsao da arvore 2: {previsao_arvore2}")
print(f"Previsao da arvore 3: {previsao_arvore3}\n")

print("Probabilidades: ")
print("Probabilidades da arvore 1")
mostrar_probabilidades(arvore1,classes,infos_usuario)
print("Probabilidades da arvore 2:")
mostrar_probabilidades(arvore2,classes,infos_usuario)
print("Probabilidades da arvore 3")
mostrar_probabilidades(arvore3,classes,infos_usuario)