"""Treina a ResNet-101 de cinco classes do olhar.

Sucessor de ``ai_train/modeltorch.py``, com quatro correções sobre o original:

- ``pretrained=True`` foi removido do torchvision; aqui se usa ``weights=``.
- O original salvava a **última** época; este salva a de melhor acurácia de
  validação, que nem sempre é a última.
- As métricas saem também por classe, não só ponderadas — sem isso, um eixo
  fraco desaparece dentro da média.
- Semente fixa e ``AMP`` opcional, para reprodutibilidade e velocidade.

Uso::

    python scripts/treinar.py --dataset data/dataset --epocas 35
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyeteractive.cnn import CLASSES, DESVIO_IMAGENET, MEDIA_IMAGENET, TAMANHO_ENTRADA


def fixar_semente(semente: int) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--saida", type=Path, default=Path("models/resnet101_model.pth"))
    parser.add_argument("--epocas", type=int, default=35)
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--taxa-aprendizado", type=float, default=1e-3)
    parser.add_argument("--trabalhadores", type=int, default=4)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="precisão mista (mais rápido em GPU)")
    parser.add_argument(
        "--limite-gpu",
        type=float,
        default=None,
        help="fração máxima da VRAM (ex.: 0.45), para conviver com outros trabalhos",
    )
    args = parser.parse_args()

    fixar_semente(args.semente)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dispositivo: {dispositivo}")

    if args.limite_gpu is not None and dispositivo.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.limite_gpu, dispositivo.index or 0)
        total = torch.cuda.get_device_properties(dispositivo).total_memory / 2**30
        print(f"teto de VRAM: {args.limite_gpu:.0%} de {total:.1f} GiB")

    transformacao = transforms.Compose([
        transforms.Resize(TAMANHO_ENTRADA),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEDIA_IMAGENET, std=DESVIO_IMAGENET),
    ])

    def carregar(split: str) -> ImageFolder:
        # Numa divisão por sessão uma classe pode ficar sem amostras de
        # validação (é o caso de 'down', que só existe numa gravação). O
        # diretório vazio precisa continuar existindo para não deslocar a
        # ordem das classes e invalidar os pesos.
        try:
            return ImageFolder(
                root=str(args.dataset / split), transform=transformacao, allow_empty=True
            )
        except TypeError:  # torchvision < 0.19 não tem allow_empty
            return ImageFolder(root=str(args.dataset / split), transform=transformacao)

    treino = carregar("train")
    validacao = carregar("val")

    if tuple(treino.classes) != CLASSES:
        raise SystemExit(
            f"Ordem de classes divergente.\n  esperada: {CLASSES}\n  encontrada: {tuple(treino.classes)}\n"
            "Os pesos salvos ficariam incompatíveis com eyeteractive.cnn."
        )

    carregador_treino = DataLoader(
        treino, batch_size=args.lote, shuffle=True,
        num_workers=args.trabalhadores, pin_memory=True,
    )
    carregador_val = DataLoader(
        validacao, batch_size=args.lote, shuffle=False,
        num_workers=args.trabalhadores, pin_memory=True,
    )
    print(f"treino: {len(treino)} imagens | validação: {len(validacao)} imagens")

    modelo = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    modelo.fc = nn.Linear(modelo.fc.in_features, len(CLASSES))
    modelo = modelo.to(dispositivo)

    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=args.taxa_aprendizado)
    escalonador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=args.epocas)
    escala = torch.amp.GradScaler("cuda", enabled=args.amp and dispositivo.type == "cuda")

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    melhor_acuracia = 0.0
    historico = []

    for epoca in range(1, args.epocas + 1):
        inicio = time.monotonic()
        modelo.train()
        perda_acumulada = 0.0

        for imagens, rotulos in carregador_treino:
            imagens = imagens.to(dispositivo, non_blocking=True)
            rotulos = rotulos.to(dispositivo, non_blocking=True)

            otimizador.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=escala.is_enabled()):
                perda = criterio(modelo(imagens), rotulos)
            escala.scale(perda).backward()
            escala.step(otimizador)
            escala.update()
            perda_acumulada += perda.item()

        escalonador.step()

        modelo.eval()
        verdadeiros, previstos = [], []
        with torch.inference_mode():
            for imagens, rotulos in carregador_val:
                imagens = imagens.to(dispositivo, non_blocking=True)
                saidas = modelo(imagens)
                previstos.extend(saidas.argmax(dim=1).cpu().numpy())
                verdadeiros.extend(rotulos.numpy())

        acuracia = float(np.mean(np.array(verdadeiros) == np.array(previstos)) * 100)
        registro = {
            "epoca": epoca,
            "perda": perda_acumulada / len(carregador_treino),
            "acuracia": acuracia,
            "precisao": float(precision_score(verdadeiros, previstos, average="weighted", zero_division=0)),
            "revocacao": float(recall_score(verdadeiros, previstos, average="weighted", zero_division=0)),
            "f1": float(f1_score(verdadeiros, previstos, average="weighted", zero_division=0)),
            "segundos": round(time.monotonic() - inicio, 1),
        }
        historico.append(registro)

        marca = ""
        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            torch.save(modelo.state_dict(), args.saida)
            marca = "  ← melhor, salvo"

        print(
            f"época {epoca:>3}/{args.epocas}  perda={registro['perda']:.4f}  "
            f"acurácia={acuracia:.2f}%  f1={registro['f1']:.4f}  "
            f"{registro['segundos']:.0f}s{marca}"
        )

    print(f"\nmelhor acurácia de validação: {melhor_acuracia:.2f}%")
    print(f"pesos salvos em: {args.saida}\n")
    print(classification_report(verdadeiros, previstos, target_names=list(CLASSES), digits=4))

    caminho_historico = args.saida.with_suffix(".historico.json")
    caminho_historico.write_text(
        json.dumps(
            {"classes": list(CLASSES), "melhor_acuracia": melhor_acuracia, "epocas": historico},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"histórico salvo em: {caminho_historico}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
