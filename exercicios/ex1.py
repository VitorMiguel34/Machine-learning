from sklearn import linear_model
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "x": [1,2,3,4,5],
    "y": [3,5,7,9,11]
})

model = linear_model.LinearRegression()
X = np.array([[x] for x in df["x"]])
y = np.array([[c] for c in df["y"]])

model.fit(X,y)

print(f"Coeficiente: {model.coef_[0]}")
print(f"Interceptor: {model.intercept_[0]}")

print(model.predict([[6]]))

#Calculando a media da soma dos erros quadraticos(MSE)

previsoes = model.predict(X)
mse = sum(y-previsoes)/(X.shape[0])
print(f"MSE: {mse}")
