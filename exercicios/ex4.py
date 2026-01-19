from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "idade": [18,22,25,30,35,40],
    "renda": [1,2,2,3,3,4],
    "compra": [0,0,1,1,1,0]
})

features = ["idade","renda"]
target = "compra"

X = df[features]
y = df[target]

model = tree.DecisionTreeClassifier(random_state=1, criterion="gini")
model.fit(X,y)

print(model.get_depth())

plt.figure(figsize=(40,40), dpi=100)

tree.plot_tree(decision_tree=model,fontsize=6,feature_names=features, class_names=["0","1"])

plt.show()