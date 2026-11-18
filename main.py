"""Trabalho de IA: MNIST com Naive Bayes e KNN.

Executar na pasta do projeto:
    python main.py validar
    python main.py finalizar

O conjunto de teste só é acessado no comando 'finalizar'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"
CONFIGURACAO = RESULTADOS / "melhores_modelos.json"
CLASSES = np.arange(10)

# É possível ampliar a busca, mas KNN em 784 dimensões pode ser demorado.
CANDIDATOS = {
    "NB_Gaussiano": {"algoritmo": "NB", "tipo": "gaussiano"},
    "NB_Multinomial": {"algoritmo": "NB", "tipo": "multinomial"},
    "KNN_k3_Euclidiana": {"algoritmo": "KNN", "k": 3, "distancia": "euclidean"},
    "KNN_k5_Euclidiana": {"algoritmo": "KNN", "k": 5, "distancia": "euclidean"},
    "KNN_k3_Manhattan": {"algoritmo": "KNN", "k": 3, "distancia": "manhattan"},
    "KNN_k5_Manhattan": {"algoritmo": "KNN", "k": 5, "distancia": "manhattan"},
}


def carregar_csv(caminho: Path) -> tuple[np.ndarray, np.ndarray]:
    """O MNIST fornecido tem 785 colunas: rótulo, seguido de 784 pixels."""
    if not caminho.exists():
        raise FileNotFoundError(f"Não encontrei {caminho}. Extraia o ZIP em 'dados/'.")
    # Os valores vão de 0 a 255; uint8 economiza memória e não altera os pixels.
    tabela = pd.read_csv(caminho, header=None, dtype=np.uint8)
    if tabela.shape[1] != 785:
        raise ValueError(f"Esperava 785 colunas em {caminho.name}; encontrei {tabela.shape[1]}.")
    y = tabela.iloc[:, 0].to_numpy(copy=True)
    X = tabela.iloc[:, 1:].to_numpy(copy=True)
    return X, y


def criar_modelo(config: dict):
    if config["algoritmo"] == "NB":
        if config["tipo"] == "gaussiano":
            return GaussianNB()
        return MultinomialNB()
    return KNeighborsClassifier(
        n_neighbors=config["k"], metric=config["distancia"],
        algorithm="brute", n_jobs=-1
    )


def taxa_erro(y_real: np.ndarray, y_previsto: np.ndarray) -> float:
    return float(np.mean(y_real != y_previsto))


def plotar_comparacao(validacao: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ordenado = validacao.sort_values("erro_validacao")
    ax.barh(ordenado["modelo"], ordenado["erro_validacao"] * 100)
    ax.invert_yaxis()
    ax.set_xlabel("Taxa de erro na validação (%)")
    ax.set_title("Comparação de configurações — sem usar o teste")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "comparacao_validacao.png", dpi=160)
    plt.close(fig)


def validar() -> None:
    RESULTADOS.mkdir(exist_ok=True)
    X, y = carregar_csv(DADOS / "mnist_train.csv")
    print(f"Treinamento original: {X.shape}; rótulos: {y.shape}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Treino interno: {len(y_tr)} | Validação: {len(y_val)}")
    linhas = []

    for nome, parametros in CANDIDATOS.items():
        print(f"Validando {nome}...", flush=True)
        modelo = criar_modelo(parametros)
        modelo.fit(X_tr, y_tr)
        pred = modelo.predict(X_val)
        erro = taxa_erro(y_val, pred)
        print(f"  Erro de validação: {erro:.2%}", flush=True)
        linhas.append({"modelo": nome, "algoritmo": parametros["algoritmo"],
                       "erro_validacao": erro})

    tabela = pd.DataFrame(linhas).sort_values("erro_validacao")
    tabela.to_csv(RESULTADOS / "validacao.csv", index=False)
    plotar_comparacao(tabela)

    melhores = {}
    for algoritmo in ("NB", "KNN"):
        melhor = tabela[tabela["algoritmo"] == algoritmo].iloc[0]
        nome = str(melhor["modelo"])
        melhores[algoritmo] = {"nome": nome, "parametros": CANDIDATOS[nome],
                              "erro_validacao": float(melhor["erro_validacao"])}
        print(f"Melhor {algoritmo}: {nome} (erro {melhor['erro_validacao']:.2%})")

    CONFIGURACAO.write_text(json.dumps(melhores, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print("\nValidação concluída. Confira os resultados e só então execute: python main.py finalizar")


def plotar_matriz(matriz: np.ndarray, nome: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    imagem = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(imagem, ax=ax, label="Quantidade de imagens")
    ax.set_xticks(CLASSES)
    ax.set_yticks(CLASSES)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title(f"Matriz de confusão — {nome}")
    fig.tight_layout()
    fig.savefig(RESULTADOS / f"matriz_{nome}.png", dpi=160)
    plt.close(fig)


def plotar_medias(X: np.ndarray, y: np.ndarray) -> None:
    """Visualização extra: médias de cada classe (sem alterar os dados dos modelos)."""
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for digito, ax in enumerate(axes.flat):
        imagem = X[y == digito].mean(axis=0).reshape(28, 28)
        ax.imshow(imagem, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Dígito {digito}")
        ax.axis("off")
    fig.suptitle("Imagem média de cada dígito — treinamento")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "medias_por_classe.png", dpi=160)
    plt.close(fig)


def finalizar() -> None:
    if not CONFIGURACAO.exists():
        raise FileNotFoundError("Execute 'python main.py validar' antes de finalizar.")
    if (RESULTADOS / "erros_teste.csv").exists():
        raise RuntimeError("O teste final já foi gerado. Não o reutilize para ajustes!")

    melhores = json.loads(CONFIGURACAO.read_text(encoding="utf-8"))
    X, y = carregar_csv(DADOS / "mnist_train.csv")
    X_teste, y_teste = carregar_csv(DADOS / "mnist_test.csv")
    plotar_medias(X, y)

    resultados = []
    for algoritmo in ("NB", "KNN"):
        nome = melhores[algoritmo]["nome"]
        modelo = criar_modelo(melhores[algoritmo]["parametros"])
        print(f"Treinando {nome} nos 60.000 exemplos...", flush=True)
        modelo.fit(X, y)
        print(f"Avaliando {nome} no teste final...", flush=True)
        pred = modelo.predict(X_teste)
        matriz = confusion_matrix(y_teste, pred, labels=CLASSES)
        pd.DataFrame(matriz, index=CLASSES, columns=CLASSES).to_csv(
            RESULTADOS / f"matriz_{nome}.csv", index_label="classe_verdadeira"
        )
        plotar_matriz(matriz, nome)

        totais = matriz.sum(axis=1)
        erros = totais - np.diag(matriz)
        erros_por_classe = erros / totais
        erro_geral = erros.sum() / matriz.sum()
        linha = {"modelo": nome, **{f"erro_{i}": erros_por_classe[i] for i in CLASSES},
                 "erro_geral": erro_geral}
        resultados.append(linha)
        print(f"Taxa de erro final {nome}: {erro_geral:.2%}")

    pd.DataFrame(resultados).to_csv(RESULTADOS / "erros_teste.csv", index=False)
    print("Resultados finais salvos na pasta 'resultados/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimento MNIST: Naive Bayes e KNN")
    parser.add_argument("etapa", choices=["validar", "finalizar"])
    args = parser.parse_args()
    if args.etapa == "validar":
        validar()
    else:
        finalizar()