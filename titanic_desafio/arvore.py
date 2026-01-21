from sklearn import tree
import pandas as pd

df = pd.read_csv("dados/train.csv")
df.dropna(inplace=True)

target = "Survived"
features = ["PassengerId","Pclass","Age","Sex","SibSp","Fare"]

df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)

X = df[features]
y = df[[target]]

model = tree.ExtraTreeClassifier()
model.fit(X,y)

test_df = pd.read_csv("dados/test.csv")
test_df["Sex"] = test_df["Sex"].apply(lambda x: 1 if x == "male" else 0)
X_test = test_df[features]
model_predict = model.predict(X_test)

results_df = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": model_predict})
results_df.to_csv("results_arvore.csv", index=False)