# Algoritmo de Regressāo Linear - Explicaçāo
---

## O que é Regressão Linear?

A regressão linear é um modelo que busca descrever a relação entre um resultado e suas variáveis, ajustando uma reta (ou hiperplano) aos dados.

Em sua forma mais simples (1 variável):

\[
y = wx + b
\]

Onde:
- `x` → variável de entrada (feature)
- `y` → valor previsto
- `a` → peso (inclinação da reta)
- `b` → bias (intercepto)

O objetivo do modelo é encontrar os valores de `a` e `b` que melhor se ajustam aos dados.

## Interpretação geométrica

Do ponto de vista geométrico, cada valor é um ponto. O algoritmo de regressão linear tenta
achar a reta que fique o mais próxima possível desses pontos. Em caso de mais de uma variável,
o algoritmo busca o hiperplano que mais se aproxime desses pontos.

---

## 📉 Função de erro (MSE)

Para saber se uma reta é boa, precisamos medir o erro.
No modelo de regressão linear, esse erro corresponde à diferença entre o valor real e o valor previsto pelo modelo.
O erro quadrático simplesmente pega o erro e o eleva ao quadrado. Logo, concluímos que, para maximizar o desempenho do modelo, precisamos minimizar a função que indica o erro.

A função usada é a **Soma dos Erros Quadráticos**, que é a soma dos erros quadráticos de todos os valores, descrita pela seguinte função:

$$
SSE = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
$$

ou

$$
SSE = \sum_{i=1}^{n} \left( y_i - (a x_i + b) \right)^2
$$

Onde:

$$
y_i \quad \text{valor real}
$$

$$
\hat{y}_i \quad \text{valor previsto pelo modelo}
$$

### Por que erro ao quadrado?
Utilizamos o erro quadrático na função porque ele penaliza erros grandes, dando uma boa estimativa do desempenho do modelo.

---

## Como minimizar essa função?

Para minimizar a Soma dos Erros Quadráticos (SSE), utilizamos **derivadas parciais** para identificar o ponto em que a função de erro atinge seu valor mínimo.
Como a função de erro depende dos parâmetros do modelo (`a` e `b`), precisamos analisar como o erro varia em relação a cada um deles separadamente.

Como o erro depende de mais de uma variável (`a` e `b`), usamos **derivadas parciais** para medir:

- Como o erro muda quando variamos `a` (mantendo `b` fixo)
- Como o erro muda quando variamos `b` (mantendo `a` fixo)

Essas derivadas indicam a direção de maior crescimento do erro.  
Para minimizar, seguimos a direção oposta.

---

## Derivadas parciais

Derivada parcial em relação a `a`:

$$
\frac{\partial \sum_{i=1}^{n} (y_i - (a x_i + b))^2}{\partial a}
$$

Derivada parcial em relação a `b`:

$$
\frac{\partial \sum_{i=1}^{n} (y_i - (a x_i + b))^2}{\partial b}
$$

Em um modelo com n variáveis, teremos n + 1 parâmetros que, ao calcular as derivadas parciais da soma dos erros quadráticos em relação a cada um deles, formam um sistema (n + 1) × (n + 1) para resolver.
Ao resolver o sistema, encontraremos o valor dos parâmetros que minimizam a soma dos erros quadráticos.

**OBS**:Como o modelo de regressāo linear prevê resultados de acordo com uma reta, sempre podemos encontrar uma reta mais distante dos valores dados em treinamento, ou seja, sempre podemos encontrar uma reta cuja soma dos erros quadráticos seja maior que a reta anterior, o que significa que as derivadas parciais indicam a reta que minimiza a SSE, nāo a que maximiza
