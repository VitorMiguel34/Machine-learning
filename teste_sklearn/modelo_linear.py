from sklearn import linear_model, tree
import pandas as pd
import matplotlib.pyplot as plt

model = linear_model.LinearRegression()
arvore1 = tree.DecisionTreeRegressor(random_state=1)
arvore2 = tree.DecisionTreeRegressor(random_state=1, max_depth=2)

df = pd.read_csv("notas.csv")

target = "nota"
X = df[["horas_de_estudo"]]
y = df[target]

model.fit(X,y)
arvore1.fit(X,y)
arvore2.fit(X,y)

predict = model.predict(X)
predict_arvore1 = arvore1.predict(X)
predict_arvore2 = arvore2.predict(X)
df["predict"] = predict

plt.plot(X,y,"o")
plt.grid(True)
plt.title("Relacao horas de estudo x nota")
plt.xlabel("Horas de estudo")
plt.ylabel("Cerveja")

plt.plot(X,predict)
plt.plot(X,predict_arvore1)
plt.plot(X,predict_arvore2)

plt.show()

plt.figure(dpi=100)

tree.plot_tree(arvore1, feature_names=["nota"], filled=True, fontsize=5)

plt.show()