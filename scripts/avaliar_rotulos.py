"""Avalia o pipeline contra vídeos rotulados por ``rotular_video.py``.

É a avaliação mais próxima do uso real disponível: vídeo contínuo, no mesmo
enquadramento da captura de produção, com rótulo por quadro. Reporta acerto
por eixo e por direção, e separa os erros em duas categorias que pedem
respostas diferentes:

- **confusão** — o sistema decidiu, e decidiu errado;
- **omissão** — o sistema respondeu "centro" onde havia direção, tipicamente
  por evidência insuficiente ou contraditória.

Em tecnologia assistiva as duas não são equivalentes: uma omissão custa uma
repetição, uma confusão dispara o comando errado.

Uso::

    python scripts/avaliar_rotulos.py --rotulos data/rotulos
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyeteractive.calibracao import carregar_perfil
from eyeteractive.fusion import ConfigFusao, Direcao
from eyeteractive.pipeline import PipelineOlhar

EIXOS = {d.value: tuple(e.value for e in d.eixos()) for d in Direcao}


def carregar_rotulos(caminho: Path) -> tuple[str, dict[int, str]]:
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    por_quadro: dict[int, str] = {}
    for segmento in dados["segmentos"]:
        for i in range(segmento["inicio"], segmento["fim"] + 1):
            por_quadro[i] = segmento["direcao"]
    return dados["video"], por_quadro


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotulos", type=Path, default=Path("data/rotulos"))
    parser.add_argument("--videos", type=Path, nargs="+",
                        default=[Path("data/videos"), Path("data/videos_fonte")])
    parser.add_argument("--pesos", type=Path, default=Path("models/resnet101_model.pth"))
    parser.add_argument("--landmarker", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--perfil", type=Path, default=None, help="perfil calibrado")
    parser.add_argument("--limite-gpu", type=float, default=0.25)
    args = parser.parse_args()

    arquivos = sorted(args.rotulos.glob("*.json")) if args.rotulos.is_dir() else []
    if not arquivos:
        print(
            f"nenhum rótulo em {args.rotulos}.\n"
            "Gere com: python scripts/rotular_video.py data/videos/<video>.mp4",
            file=sys.stderr,
        )
        return 1

    config = carregar_perfil(args.perfil) if args.perfil else ConfigFusao()
    if args.perfil:
        print(f"usando perfil calibrado: {args.perfil}")

    acertos_v = acertos_h = acertos_dir = 0
    total = 0
    confusao = defaultdict(Counter)
    erros = Counter()
    sem_rosto = 0

    with PipelineOlhar(
        pesos_cnn=args.pesos,
        modelo_landmarker=args.landmarker,
        config=config,
        limite_memoria_gpu=args.limite_gpu,
    ) as pipeline:
        for arquivo in arquivos:
            nome, por_quadro = carregar_rotulos(arquivo)
            caminho = next(
                (p / nome for p in args.videos if (p / nome).is_file()), None
            )
            if caminho is None:
                print(f"aviso: vídeo {nome} não encontrado, pulando")
                continue

            pipeline.reiniciar()
            captura = cv2.VideoCapture(str(caminho))
            fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
            indice = 0
            usados = 0

            while True:
                ok, quadro = captura.read()
                if not ok:
                    break
                esperado = por_quadro.get(indice)
                indice += 1
                if esperado is None:
                    continue

                analise = pipeline.processar(quadro, timestamp_ms=int(indice * 1000 / fps))
                if not analise.rosto_detectado:
                    sem_rosto += 1
                    continue

                v_esperado, h_esperado = EIXOS[esperado]
                resultado = analise.resultado
                total += 1
                usados += 1
                acertos_v += resultado.vertical.valor == v_esperado
                acertos_h += resultado.horizontal.valor == h_esperado
                obtido = resultado.direcao.value
                acertos_dir += obtido == esperado
                confusao[esperado][obtido] += 1

                if obtido != esperado:
                    erros["omissão" if obtido == "centro" else "confusão"] += 1

            captura.release()
            print(f"{nome:<18} {usados:>5} quadros avaliados")

    if total == 0:
        print("erro: nenhum quadro avaliado", file=sys.stderr)
        return 1

    print(f"\nquadros avaliados: {total}   sem rosto: {sem_rosto}")
    print(f"\nacerto do eixo vertical:   {acertos_v / total * 100:6.2f}%")
    print(f"acerto do eixo horizontal: {acertos_h / total * 100:6.2f}%")
    print(f"acerto das 9 direções:     {acertos_dir / total * 100:6.2f}%")

    if erros:
        soma = sum(erros.values())
        print(f"\nerros: {soma}")
        for tipo, n in erros.most_common():
            print(f"  {tipo:<10} {n:>5}  ({n / soma * 100:.1f}% dos erros)")

    print("\nmatriz de confusão (linha = rótulo, coluna = decisão)")
    rotulos_vistos = [d for d in EIXOS if d in confusao]
    colunas = sorted({c for linha in confusao.values() for c in linha})
    largura = max(14, max((len(c) for c in colunas), default=8) + 1)
    print(" " * 16 + "".join(f"{c:>{largura}}" for c in colunas))
    for rotulo in rotulos_vistos:
        celulas = "".join(f"{confusao[rotulo][c]:>{largura}}" for c in colunas)
        print(f"{rotulo:<16}{celulas}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
