# MNIST — Naive Bayes e KNN

Projeto de classificação de dígitos manuscritos utilizando os algoritmos **Naive Bayes** e **K-Nearest Neighbors (KNN)**, com o dataset MNIST.

O objetivo é comparar o desempenho dos dois classificadores e avaliar diferentes configurações.

## Dataset

O MNIST contém 70.000 imagens de dígitos de 0 a 9, com resolução de 28×28 pixels:

- 60.000 imagens para treinamento.
- 10.000 imagens para teste.

## Metodologia

O conjunto de treinamento foi dividido utilizando **Holdout estratificado**:

- 80% para treinamento (48.000 imagens).
- 20% para validação (12.000 imagens).

Foram testadas duas distribuições para o Naive Bayes (Gaussiana e Multinomial) e diferentes configurações de KNN (K=3 e K=5, com distâncias Euclidiana e Manhattan).

As melhores configurações foram selecionadas com base na taxa de erro de validação. Em seguida, os classificadores foram treinados com os 60.000 exemplos e avaliados no conjunto de teste.

Não foi realizado pré-processamento dos pixels.

## Resultados

| Modelo | Erro na validação | Erro no teste |
|---|---:|---:|
| Naive Bayes Multinomial | 17,62% | 16,35% |
| KNN (K=3, Euclidiana) | 2,97% | 2,95% |

O KNN apresentou o melhor desempenho, alcançando **97,05% de acurácia** no conjunto de teste.

As matrizes de confusão, taxas de erro por classe e visualizações estão disponíveis na pasta `resultados/`.

## Tecnologias

Python, NumPy, Pandas, Scikit-learn e Matplotlib.

## Execução

Instale as dependências:

```bash
pip install numpy pandas scikit-learn matplotlib
```

Coloque os arquivos `mnist_train.csv` e `mnist_test.csv` na pasta `dados/`.

Execute a validação:

```bash
python main.py validar
```

Execute a avaliação final:

```bash
python main.py finalizar
```

O conjunto de teste é utilizado somente na avaliação final. O programa impede repetir essa etapa caso o arquivo de resultados finais já exista.