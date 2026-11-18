"""Gera as figuras do relatório e da apresentação a partir dos CSVs em resultados/.

Não treina modelos e não acessa mnist_test.csv: só lê arquivos já produzidos por
'python main.py validar' e 'python main.py finalizar'.

Uso (na pasta do projeto):
    python gerar_figuras.py
    python gerar_figuras.py --pasta caminho/para/resultados --saida figuras

Arquivos gerados em <saida>/:
    comparacao_validacao_relatorio.png   Figura 1 (fundo branco, barras verticais com rótulos)
    comparacao_validacao_slide.png       Gráfico do slide 6 (fundo escuro, rótulos alinhados)
    matriz_<modelo>_anotada.png          Figuras 2 e 3 (matriz com o número em cada célula)
    erro_por_classe.png                  Figura 4 (barras NB x KNN por dígito)
    maiores_confusoes.txt                Maiores confusões de cada modelo (conferir slide 7)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = np.arange(10)
NOMES_BONITOS = {
    "KNN_k3_Euclidiana": "KNN k=3 Euclidiana",
    "KNN_k5_Euclidiana": "KNN k=5 Euclidiana",
    "KNN_k3_Manhattan": "KNN k=3 Manhattan",
    "KNN_k5_Manhattan": "KNN k=5 Manhattan",
    "NB_Multinomial": "NB Multinomial",
    "NB_Gaussiano": "NB Gaussiano",
}


def fmt_pct(x: float, casas: int = 2) -> str:
    """Formata percentual com vírgula decimal e arredondamento comum (17,625 -> 17,63)."""
    from decimal import ROUND_HALF_UP, Decimal
    q = Decimal(1).scaleb(-casas)
    v = Decimal(str(round(x, 10))).quantize(q, rounding=ROUND_HALF_UP)
    return f"{v}".replace(".", ",") + "%"


def comparacao_relatorio(val: pd.DataFrame, saida: Path) -> None:
    val = val.sort_values("erro_validacao")
    erros = val["erro_validacao"].to_numpy() * 100
    rotulos = [NOMES_BONITOS[m].replace(" ", "\n", 1) for m in val["modelo"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    barras = ax.bar(range(len(erros)), erros, color="#1f77b4")
    for b, e in zip(barras, erros):
        ax.text(b.get_x() + b.get_width() / 2, e + 0.6, fmt_pct(e),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(erros)))
    ax.set_xticklabels(rotulos, fontsize=8)
    ax.set_ylabel("Erro de validação (%)")
    ax.set_title("Comparação dos modelos na validação")
    ax.set_ylim(0, erros.max() * 1.12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(saida / "comparacao_validacao_relatorio.png", dpi=200)
    plt.close(fig)


def comparacao_slide(val: pd.DataFrame, saida: Path) -> None:
    """Versão para o slide 6: fundo escuro, cada rótulo exatamente ao lado da sua barra."""
    fundo, texto, dourado, destaque = "#0a0a0a", "#d9d9d9", "#a88022", "#ffd23f"
    val = val.sort_values("erro_validacao").reset_index(drop=True)
    erros = val["erro_validacao"].to_numpy() * 100
    rotulos = [f"{NOMES_BONITOS[m]} · {e:.4f}%".replace(".", ",")
               for m, e in zip(val["modelo"], erros)]
    cores = [destaque] + [dourado] * (len(erros) - 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=fundo)
    ax.set_facecolor(fundo)
    y = np.arange(len(erros))
    ax.barh(y, erros, color=cores, height=0.62)
    ax.set_yticks(y)  # um tick por barra: garante o alinhamento rótulo <-> barra
    ax.set_yticklabels(rotulos, color=texto, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 50)
    ax.xaxis.tick_top()
    ax.set_xlabel("Erro de validação (%), 12.000 imagens", color=texto, labelpad=10)
    ax.xaxis.set_label_position("top")
    ax.tick_params(colors=texto, length=0)
    ax.grid(axis="x", color="#333333", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("right", "bottom", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["top"].set_color("#777777")
    fig.tight_layout()
    fig.savefig(saida / "comparacao_validacao_slide.png", dpi=200, facecolor=fundo)
    plt.close(fig)


def matriz_anotada(matriz: np.ndarray, nome: str, saida: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Quantidade de imagens")
    limite = matriz.max() / 2
    for i in CLASSES:
        for j in CLASSES:
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if matriz[i, j] > limite else "#333333")
    ax.set_xticks(CLASSES)
    ax.set_yticks(CLASSES)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title(f"Matriz de confusão — {NOMES_BONITOS.get(nome, nome)}")
    fig.tight_layout()
    fig.savefig(saida / f"matriz_{nome}_anotada.png", dpi=200)
    plt.close(fig)


def erro_por_classe(erros: pd.DataFrame, saida: Path) -> None:
    nb = erros[erros["modelo"] == "NB_Multinomial"].iloc[0]
    knn = erros[erros["modelo"] == "KNN_k3_Euclidiana"].iloc[0]
    cols = [f"erro_{i}" for i in CLASSES]
    larg = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(CLASSES - larg / 2, nb[cols].astype(float) * 100, larg,
           label="NB Multinomial", color="#1f77b4")
    ax.bar(CLASSES + larg / 2, knn[cols].astype(float) * 100, larg,
           label="KNN k=3 Euclidiana", color="#ff7f0e")
    ax.set_xticks(CLASSES)
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Taxa de erro (%)")
    ax.set_title("Taxa de erro por classe no conjunto de teste")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(saida / "erro_por_classe.png", dpi=200)
    plt.close(fig)


def maiores_confusoes(matrizes: dict[str, np.ndarray], saida: Path, n: int = 6) -> None:
    """Lista as maiores confusões fora da diagonal, sinalizando empates."""
    linhas = []
    for nome, m in matrizes.items():
        fora = m.copy()
        np.fill_diagonal(fora, 0)
        itens = [(int(fora[i, j]), i, j) for i in CLASSES for j in CLASSES if fora[i, j] > 0]
        itens.sort(key=lambda t: (-t[0], t[1], t[2]))
        linhas.append(f"{nome}")
        for valor, i, j in itens[:n]:
            empate = sum(1 for v, _, _ in itens if v == valor)
            marca = f"  (empate com {empate - 1} outra(s))" if empate > 1 else ""
            linhas.append(f"  {i}->{j}: {valor}{marca}")
        linhas.append("")
    (saida / "maiores_confusoes.txt").write_text("\n".join(linhas), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Gera figuras a partir dos resultados salvos")
    p.add_argument("--pasta", default="resultados", help="pasta com os CSVs de resultados")
    p.add_argument("--saida", default="figuras", help="pasta de saída das figuras")
    a = p.parse_args()
    pasta, saida = Path(a.pasta), Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(pasta / "validacao.csv")
    erros = pd.read_csv(pasta / "erros_teste.csv")
    comparacao_relatorio(val, saida)
    comparacao_slide(val, saida)
    erro_por_classe(erros, saida)

    matrizes = {}
    for nome in ("KNN_k3_Euclidiana", "NB_Multinomial"):
        m = pd.read_csv(pasta / f"matriz_{nome}.csv", index_col=0).to_numpy()
        matrizes[nome] = m
        matriz_anotada(m, nome, saida)
    maiores_confusoes(matrizes, saida)
    print(f"Figuras salvas em {saida}/")


if __name__ == "__main__":
    main()
