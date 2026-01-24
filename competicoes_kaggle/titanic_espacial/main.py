import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df_treino = pd.read_csv("train.csv")

def tratar_df(df: pd.DataFrame) -> pd.DataFrame:
    df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].value_counts().idxmax())
    df["CryoSleep"] = df["CryoSleep"].fillna(df["CryoSleep"].value_counts().idxmax()).astype(int)
    df["Cabin"] = df["Cabin"].fillna(df["Cabin"].value_counts().idxmax())
    df["Cabin"] = df["Cabin"].apply(lambda x:x[-1])
    df["Destination"] = df["Destination"].fillna(df["Destination"].value_counts().idxmax())
    df["Age"] = df["Age"].fillna(int(df["Age"].median()))
    df["VIP"] = df["VIP"].fillna(False).astype(int)
    df["Name"] = df["Name"].apply(lambda x: 1 if x else 0)

    for feature in ["RoomService","FoodCourt","ShoppingMall","VRDeck","Spa"]:
        df[feature] = df[feature].fillna(df[feature].median())
    
    return df

df_treino = tratar_df(df_treino)
features = ["CryoSleep","Age","VIP","RoomService","ShoppingMall","FoodCourt","Spa","VRDeck"]
target = "Transported"
alvos_encoder = ["HomePlanet","Destination","Cabin","Name"]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.set_output(transform="pandas")
encoder_output = encoder.fit_transform(df_treino[alvos_encoder])
df_treino = pd.concat([df_treino,encoder_output],axis=1)
for feature in list(encoder_output.columns):
    features.append(feature)

model = HistGradientBoostingClassifier(random_state=42)
X_treino = df_treino[features]
y_treino = df_treino[target]
model.fit(X_treino,y_treino)

df_teste = pd.read_csv("test.csv")
df_teste = tratar_df(df_teste)
encoder_test_output = encoder.fit_transform(df_teste[alvos_encoder])
df_teste = pd.concat([df_teste,encoder_test_output],axis=1)

X_teste = df_teste[features]
previsao = model.predict(X_teste)
df_resultado = pd.DataFrame({
    "PassengerId": [x for x in df_teste["PassengerId"]],
    "Transported": [x for x in previsao] 
})
df_resultado.to_csv("resultado3.csv", index=False)
print(f"Acurácia no treino: {accuracy_score(y_treino,model.predict(X_treino))}")