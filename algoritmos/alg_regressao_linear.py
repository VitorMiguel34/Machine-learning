import sympy as sp
import pandas as pd

df = pd.read_csv("dados.csv")
args_num = int(df.shape[0])
a,b = sp.symbols("a b")
x_params = [sp.symbols(f'x{c}') for c in range(1,args_num + 1)]
y_params = [sp.symbols(f'y{c}') for c in range(1,args_num + 1)]

x_values = list(df["x"])
y_values = list(df["y"])

values = {}
J = 0

for index, x_param in enumerate(x_params):
    J += (y_params[index] - (a*x_param + b))**2
    values[x_params[index]] = x_values[index]
    values[y_params[index]] = y_values[index]

dj_da = sp.diff(J,a)
dj_db = sp.diff(J,b)

eq1 = sp.Eq(dj_da,0)
eq2 = sp.Eq(dj_db,0)

solution = sp.solve([eq1,eq2],(a,b), simplify=True)

a_real = float(solution[a].subs(values))
b_real = float(solution[b].subs(values))

print(f"a = {a_real}")
print(f"b = {b_real}")

min_error = J.subs(values).subs({a: a_real, b: b_real})
print(f"Erro minimo: {min_error}")

horas = int(input("Quantidade de horas estudadas:"))
calcular_nota = lambda x:a_real*x + b_real
nota = calcular_nota(horas)
print(f"Voce tirou {nota}!")
