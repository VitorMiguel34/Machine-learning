from sklearn import linear_model
import pandas as pd

df = pd.read_csv("dados/train.csv")
df.dropna(inplace=True)

target = "Survived"
features = ["PassengerId","Pclass","Age","Sex","SibSp","Fare"]

df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)

X = df[features]
y = df[[target]]

model = linear_model.LinearRegression()
model.fit(X,y)

test_df = pd.read_csv("dados/test.csv")
test_df["Sex"] = test_df["Sex"].apply(lambda x: 1 if x == "male" else 0)

for feature in features:
    test_df[feature].fillna(test_df[feature].median(), inplace=True)

X_test = test_df[features]

for feature in features:
    X_test[feature].fillna(X_test[feature].median())
print(X_test)

model_predict = [1 if x[0] >= 1 else 0 for x in model.predict(X_test)]
print(model_predict)

results_df = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": model_predict}
)
print(results_df)
results_df.to_csv("results_regressao_linear.csv", index=False)