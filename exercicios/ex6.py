import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd

'''O objetivo desse exercício é comparar os três critêrios de uma árvore de decisāo(classificaçāo):
-gini
-entropia
-log loss

Para fazer tal comparaçāo, simularemos a seguinte situaçāo: 
Qureremos prever a faixa etária de um indivíduo com base em 3 informações:
1.Se a pessoa estuda: 1 se estuda e 0 se nāo estuda
2.Se a pessoa trabalha: 1 se trabalha e 0 se nāo trabalha
3.Estado civil: 1 para casado e 0 para solteiro

Classificações: Jovem, Adulto e Idoso

OBS:Consulte a pasta de dados para visualizar o csv usado no exercicio e a pasta resultados 
para consultar os plots realizados
'''

df = pd.read_csv("dados/ex6.csv")

features = ["estuda","trabalha","estado_civil"]
target = "faixa_etaria"
classes = ["Jovem","Adulto","Idoso"]

X = df[features]
y = df[target]

#Arvore com o criterio gini
arvore1 = tree.DecisionTreeClassifier(criterion="gini")
#Avore com o criterio entropia
arvore2 = tree.DecisionTreeClassifier(criterion="entropy")
#Avore com o criterio log loss
arvore3 = tree.DecisionTreeClassifier(criterion="log_loss")

arvore1.fit(X,y)
arvore2.fit(X,y)
arvore3.fit(X,y)

def plotar_arvores():
    #Plotando a arvore 1(gini)
    plt.figure(figsize=(40,40))
    tree.plot_tree(decision_tree=arvore1,feature_names=features, class_names=classes, fontsize=7)
    plt.show()

    #Plotando a arvore 2(entropia)
    plt.figure(figsize=(40,40))
    tree.plot_tree(decision_tree=arvore2,feature_names=features,class_names=classes, fontsize=7)
    plt.show()

    #Plotando a arvore 3(log loss)
    plt.figure(figsize=(40,40))
    tree.plot_tree(decision_tree=arvore3,feature_names=features, class_names=classes,fontsize=7)
    plt.show()


if __name__ == "__main__":
    plotar_arvores()
