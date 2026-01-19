from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#O objetivo é comparar árvores com diferentes níveis de profundidade

df = pd.DataFrame({
    "idade": [18,22,25,30,35,40],
    "renda": [1,2,2,3,3,4],
    "compra": [0,0,1,1,1,0]
})

features = ["idade","renda"]
target = "compra"

X = df[features]
y = df[target]

arvore1 = tree.DecisionTreeClassifier(random_state=1, criterion="gini")
arvore2 = tree.DecisionTreeClassifier(random_state=1, criterion="gini", max_depth=1)

arvore1.fit(X,y)
arvore2.fit(X,y)

plt.figure(figsize=(40,40))

tree.plot_tree(arvore1, feature_names=features, class_names=["0","1"], fontsize=7)

plt.show()


plt.figure(figsize=(40,40))

tree.plot_tree(arvore2, feature_names=features, class_names=["0","1"], fontsize=7)

plt.show()

