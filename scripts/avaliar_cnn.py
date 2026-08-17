"""Avalia a ResNet-101 no conjunto de validação, por classe e por eixo.

Reporta três coisas:

1. Acurácia das cinco classes originais — a métrica do artigo.
2. Acurácia da **marginal vertical**, que é o que passa a governar o eixo
   cima/baixo na arquitetura nova.
3. Acurácia da **marginal horizontal**, para dimensionar o peso da CNN como
   segunda fonte do eixo que a geometria já resolve.

Uso::

    python scripts/avaliar_cnn.py --dataset data/dataset/val
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyeteractive.cnn import CLASSES, ClassificadorOlhar

# Eixo esperado para cada classe do dataset.
VERTICAL_ESPERADO = {"up": "cima", "down": "baixo", "center": "centro", "left": "centro", "right": "centro"}
HORIZONTAL_ESPERADO = {"left": "esquerda", "right": "direita", "center": "centro", "up": "centro", "down": "centro"}


def decidir_vertical(m) -> str:
    return max((("cima", m.cima), ("baixo", m.baixo), ("centro", m.centro_vertical)), key=lambda x: x[1])[0]


def decidir_horizontal(m) -> str:
    return max(
        (("esquerda", m.esquerda), ("direita", m.direita), ("centro", m.centro_horizontal)),
        key=lambda x: x[1],
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset/val"))
    parser.add_argument("--pesos", type=Path, default=Path("models/resnet101_model.pth"))
    parser.add_argument("--lote", type=int, default=64)
    parser.add_argument("--limite-por-classe", type=int, default=0, help="0 = todas")
    parser.add_argument("--limite-gpu", type=float, default=0.25)
    args = parser.parse_args()

    classificador = ClassificadorOlhar(args.pesos, limite_memoria_gpu=args.limite_gpu)
    print(f"dispositivo: {classificador.dispositivo}\n")

    acertos_classe = Counter()
    total_classe = Counter()
    acertos_vertical = Counter()
    acertos_horizontal = Counter()
    confusao = defaultdict(Counter)

    for classe in CLASSES:
        pasta = args.dataset / classe
        arquivos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if args.limite_por_classe:
            arquivos = arquivos[: args.limite_por_classe]

        for inicio in range(0, len(arquivos), args.lote):
            bloco = arquivos[inicio : inicio + args.lote]
            recortes = []
            validos = []
            for caminho in bloco:
                img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                recortes.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                validos.append(caminho)

            for marginais in classificador.prever_lote(recortes):
                total_classe[classe] += 1
                previsto = marginais.classe_argmax
                confusao[classe][previsto] += 1
                if previsto == classe:
                    acertos_classe[classe] += 1
                if decidir_vertical(marginais) == VERTICAL_ESPERADO[classe]:
                    acertos_vertical[classe] += 1
                if decidir_horizontal(marginais) == HORIZONTAL_ESPERADO[classe]:
                    acertos_horizontal[classe] += 1

    total = sum(total_classe.values())
    if total == 0:
        print("erro: nenhuma imagem avaliada", file=sys.stderr)
        return 1

    print(f"{'classe':<10}{'n':>6}{'acurácia 5-cls':>16}{'marg. vertical':>16}{'marg. horizontal':>18}")
    print("-" * 66)
    for classe in CLASSES:
        n = total_classe[classe]
        if not n:
            continue
        print(
            f"{classe:<10}{n:>6}"
            f"{acertos_classe[classe] / n * 100:>15.2f}%"
            f"{acertos_vertical[classe] / n * 100:>15.2f}%"
            f"{acertos_horizontal[classe] / n * 100:>17.2f}%"
        )
    print("-" * 66)
    print(
        f"{'TOTAL':<10}{total:>6}"
        f"{sum(acertos_classe.values()) / total * 100:>15.2f}%"
        f"{sum(acertos_vertical.values()) / total * 100:>15.2f}%"
        f"{sum(acertos_horizontal.values()) / total * 100:>17.2f}%"
    )

    print("\nmatriz de confusão (linha = verdadeiro, coluna = previsto)")
    print(f"{'':<10}" + "".join(f"{c:>9}" for c in CLASSES))
    for verdadeiro in CLASSES:
        linha = "".join(f"{confusao[verdadeiro][p]:>9}" for p in CLASSES)
        print(f"{verdadeiro:<10}{linha}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
