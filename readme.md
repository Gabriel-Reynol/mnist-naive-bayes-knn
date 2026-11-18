# Classificação de Dígitos Manuscritos MNIST — Naive Bayes e KNN

## 1. Objetivo

Este projeto implementa e compara os classificadores Naive Bayes (NB) e K-Nearest Neighbors (KNN) na tarefa de reconhecimento de dígitos manuscritos utilizando o conjunto de dados MNIST.

O objetivo é avaliar diferentes configurações dos algoritmos, selecionar os melhores modelos por meio de validação e comparar seu desempenho no conjunto de teste.

## 2. Conjunto de dados

O MNIST contém 70.000 imagens em escala de cinza, representando dígitos de 0 a 9, com resolução de 28 × 28 pixels.

- Treinamento: 60.000 imagens.
- Teste: 10.000 imagens.
- Cada imagem é representada por 784 valores de intensidade dos pixels.
- Cada linha dos arquivos CSV contém o rótulo da imagem seguido dos 784 pixels.

Os arquivos utilizados são:

- `dados/mnist_train.csv`
- `dados/mnist_test.csv`

## 3. Metodologia

O conjunto original de treinamento foi dividido por meio de holdout estratificado, utilizando 80% para treinamento interno e 20% para validação.

- Treinamento interno: 48.000 imagens.
- Validação: 12.000 imagens.
- Semente aleatória: 42.

Não foi realizado pré-processamento dos pixels. Os valores originais de intensidade, entre 0 e 255, foram utilizados pelos classificadores.

### 3.1. Modelos avaliados

Foram avaliadas seis configurações:

**Naive Bayes**
- Gaussian Naive Bayes.
- Multinomial Naive Bayes.

**K-Nearest Neighbors**
- KNN com k = 3 e distância Euclidiana.
- KNN com k = 5 e distância Euclidiana.
- KNN com k = 3 e distância Manhattan.
- KNN com k = 5 e distância Manhattan.

As melhores configurações de cada algoritmo foram escolhidas com base no erro de validação.

### 3.2. Avaliação final

Após a seleção, os modelos escolhidos foram treinados novamente utilizando os 60.000 exemplos originais de treinamento.

O conjunto de teste foi utilizado exclusivamente na avaliação final, uma única vez por classificador.

## 4. Resultados

### 4.1. Resultados da validação

| Modelo | Erro de validação |
|---|---:|
| KNN k=3 Euclidiana | 2,97% |
| KNN k=5 Euclidiana | 3,25% |
| KNN k=3 Manhattan | 3,67% |
| KNN k=5 Manhattan | 3,80% |
| NB Multinomial | 17,63% |
| NB Gaussiano | 44,28% |

O KNN com k=3 e distância Euclidiana apresentou o menor erro entre as configurações avaliadas. Entre os modelos Naive Bayes, o Multinomial apresentou o melhor resultado.

### 4.2. Resultados no conjunto de teste

| Classificador | Erro | Acurácia |
|---|---:|---:|
| Naive Bayes Multinomial | 16,35% | 83,65% |
| KNN k=3 Euclidiana | 2,95% | 97,05% |

O KNN apresentou desempenho superior neste experimento, com menor taxa de erro geral.

### 4.3. Matrizes de confusão e erros por classe

As matrizes de confusão e as taxas de erro por classe estão disponíveis na pasta `resultados/`, tanto em formato CSV quanto em imagem PNG.

A análise das matrizes permite identificar quais dígitos são mais frequentemente confundidos pelos classificadores.

## 5. Extensões e visualizações

Além da comparação dos classificadores, foram desenvolvidas visualizações para explorar as características das imagens do conjunto de treinamento.

### 5.1. Comparação das configurações

O gráfico `comparacao_validacao.png` apresenta as taxas de erro dos seis modelos avaliados, permitindo visualizar a diferença de desempenho entre as configurações.

### 5.2. Imagens médias por classe

O arquivo `medias_por_classe.png` apresenta, para cada dígito, a média dos valores de cada pixel entre as imagens daquela classe.

Essa representação permite observar os padrões espaciais característicos de cada classe.

### 5.3. Variância dos pixels por classe

O arquivo `variancias_por_classe.png` apresenta a variância dos valores de cada pixel para as imagens pertencentes a cada dígito.

Regiões de maior variância indicam maior dispersão das intensidades dos pixels entre as imagens da mesma classe. Essa visualização permite explorar a variabilidade das diferentes formas de escrita dos dígitos.

As visualizações de médias e variâncias são análises exploratórias dos dados e não representam etapas de pré-processamento dos classificadores.

## 6. Estrutura do projeto

```text
Trabalho_MNIST/
├── main.py
├── readme.md
├── requirements.txt
├── generate_mnist_csv.py
├── dados/
│   ├── mnist_train.csv
│   └── mnist_test.csv
└── resultados/
    ├── validacao.csv
    ├── melhores_modelos.json
    ├── comparacao_validacao.png
    ├── erros_teste.csv
    ├── matriz_NB_Multinomial.csv
    ├── matriz_NB_Multinomial.png
    ├── matriz_KNN_k3_Euclidiana.csv
    ├── matriz_KNN_k3_Euclidiana.png
    ├── medias_por_classe.png
    └── variancias_por_classe.png
```

## 7. Tecnologias e dependências

O projeto foi desenvolvido em Python e utiliza:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Para instalar as dependências:

```bash
pip install -r requirements.txt
```

## 8. Execução

Execute os comandos a partir da pasta principal do projeto.

Primeiro, realize a validação:

```bash
python main.py validar
```

Após conferir os modelos selecionados, execute a avaliação final:

```bash
python main.py finalizar
```

A etapa de finalização gera as matrizes de confusão, os erros por classe e geral, além das visualizações de médias e variâncias.

**Atenção:** o programa impede a repetição da avaliação final quando `resultados/erros_teste.csv` já existe. Essa proteção evita reutilizar o conjunto de teste para ajustes posteriores.

## 9. Considerações finais

Os resultados obtidos indicam que, nas configurações avaliadas, o KNN com k=3 e distância Euclidiana apresentou melhor desempenho que o Naive Bayes Multinomial.

A comparação evidencia diferenças entre os métodos e permite discutir suas características, limitações e comportamento diante da variabilidade das imagens manuscritas.

As visualizações complementares ampliam a análise dos dados, oferecendo representações das médias e variâncias dos pixels para cada classe.
