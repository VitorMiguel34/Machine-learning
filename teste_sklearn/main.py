import numpy as np
import pandas as pd
import sklearn.tree as tree
import matplotlib.pyplot as plt

X = [[0],[5],[6],[10]]
y = [["Nota ruim"],["Nota ruim"],["Boa nota"],["Boa nota"]]

model = tree.DecisionTreeClassifier(random_state=1)
model.fit(X,y)

prever = float(input("Nota: "))

previsao = model.predict([[prever]])
print(previsao[0])
