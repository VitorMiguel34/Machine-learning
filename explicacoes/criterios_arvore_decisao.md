# 📊 Critérios de Impureza em Árvores de Decisão  
### Gini, Entropia e Log Loss

Em modelos de **árvores de decisão para classificação**, os critérios de impureza são usados para decidir **como dividir os dados em cada nó**. O objetivo é sempre criar subconjuntos cada vez mais “puros”, ou seja, contendo predominantemente exemplos de uma única classe.

Neste arquivo, são explorados três critérios principais:
- **Gini**
- **Entropia**
- **Log Loss**

---

## 🌳 O que é Impureza?

A **impureza** mede o quão misturadas estão as classes em um nó da árvore.

- Nó **puro** → todas as amostras pertencem à mesma classe  
- Nó **impuro** → várias classes misturadas  

Quanto **menor a impureza**, melhor é o nó.

---

## 🔹 Gini (Índice de Gini)

O **Índice de Gini** quantifica o quão misturado está um nó em termos de classes, medindo a **probabilidade de erro ao rotular uma amostra de forma aleatória**, usando a própria distribuição de classes presente no nó.

A intuição é a seguinte: imagine que você escolhe **uma amostra aleatória de um nó** e tenta prever sua classe **sorteando uma classe de acordo com as proporções existentes nesse nó**. O valor do Gini representa a **probabilidade de essa previsão estar errada**.

Dessa forma:
- Se o nó é composto quase inteiramente por uma única classe, a chance de erro é pequena → **Gini baixo**
- Se o nó possui classes bem equilibradas, a chance de erro é alta → **Gini alto**

### Fórmula

$$
Gini = 1 - \sum_{i=1}^{n} p_i^2
$$

onde:
- $p_i$ é a proporção da classe *i* no nó  
- $n$ é o número de classes

### Propriedades

- Varia de **0 a aproximadamente 0.5** (em classificação binária)
- **0** indica um nó totalmente puro

### Interpretação

- Favorece divisões que isolam rapidamente a classe majoritária

---

## 🔹 Entropia

A **Entropia**, originada da Teoria da Informação, mede o **nível de incerteza** associado à classe de uma amostra escolhida aleatoriamente em um nó da árvore.

Ela responde, de forma intuitiva, à pergunta:  
**“Quanta informação é necessária, em média, para identificar corretamente a classe de uma amostra desse nó?”**

Se o nó é quase puro, a incerteza é baixa e pouca informação é necessária.  
Se as classes estão bem distribuídas, a incerteza é alta e mais informação é necessária.

Assim, a entropia cresce à medida que as classes ficam mais misturadas e imprevisíveis.

### Fórmula

$$
Entropia = - \sum_{i=1}^{n} p_i \log_2(p_i)
$$

### Propriedades

- Varia de **0 a $\log_2(n)$**, onde *n* é o número de classes
- **0** indica um nó puro
- Penaliza mais fortemente divisões muito incertas

### Interpretação

- Quanto maior a entropia, maior a desordem
- Busca divisões que maximizem o **ganho de informação**

---

## 🔹 Log Loss (Logarithmic Loss)

O **Log Loss** avalia a qualidade das **probabilidades previstas pelo modelo**, levando em conta não apenas se a classe prevista está correta, mas também **o nível de confiança associado a essa previsão**.

Ele pode ser interpretado como a resposta à pergunta:  
**“O modelo atribuiu alta probabilidade à classe correta?”**

A ideia central é:
- Previsões corretas com **alta confiança** são pouco penalizadas
- Previsões erradas com **alta confiança** são severamente penalizadas
- Previsões incertas recebem penalizações intermediárias

Por isso, o Log Loss é especialmente útil para modelos probabilísticos, nos quais errar “com muita certeza” é considerado muito pior do que errar de forma incerta.

### Fórmula

$$
LogLoss = - \sum_{i=1}^{n} y_i \log(p_i)
$$

onde:
- $y_i = 1$ se a classe correta for *i*, caso contrário $0$
- $p_i$ é a probabilidade prevista para a classe *i*

### Propriedades

- Nunca é negativa
- Quanto menor, melhor
- Extremamente sensível a erros cometidos com alta confiança

### Interpretação

- Avalia não apenas a classe prevista, mas também **o nível de confiança** do modelo

---

## ⚖️ Comparação entre os Critérios

| Critério   | Mede o quê? | Vantagens | Observações |
|-----------|------------|----------|-------------|
| **Gini** | Impureza | Rápido e eficiente | Muito usado na prática |
| **Entropia** | Incerteza | Base teórica forte | Resultados similares ao Gini |
| **Log Loss** | Qualidade probabilística | Penaliza erros confiantes | Pode gerar árvores mais profundas |

---

## 🧠 Observações Importantes

- Em muitos casos, **Gini e Entropia produzem árvores muito parecidas**
- Um nó pode **não ser puro** e, ainda assim:
  - não se dividir mais
  - prever uma classe com probabilidade 1 (classe majoritária)
- Árvores de decisão **não são bons estimadores de probabilidade**

---

## 📌 Conclusão

Os critérios de impureza definem como a árvore aprende a separar os dados.  
Entender **Gini**, **Entropia** e **Log Loss** é essencial para interpretar corretamente o comportamento de árvores de decisão e seus limites, especialmente no que diz respeito a previsões probabilísticas.
