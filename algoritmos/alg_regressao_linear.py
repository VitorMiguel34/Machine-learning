import sympy as sp
import pandas as pd

df = pd.read_csv("dados.csv")
args_num = int(df.shape[0])
a,b = sp.symbols("a b")
x_params = [sp.symbols(f'x{c}') for c in range(1,args_num + 1)]
y_params = [sp.symbols(f'y{c}') for c in range(1,args_num + 1)]

x_values = list(df["horas_trabalhadas"])
y_values = list(df["dinheiro"])

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
if min_error == 0:
    print("Note que os dados estāo dispostos de forma completamente linear")
else:
    print("Note que os dados nāo estāo dispostos de forma completamente linear")

horas_trabalhadas = float(input("Horas trabalhadas: "))
calcular_dinheiro = lambda x:a_real*x + b_real
nota = calcular_dinheiro(horas_trabalhadas)
print(f"Dinheiro ganho: {nota:.2f}!")
