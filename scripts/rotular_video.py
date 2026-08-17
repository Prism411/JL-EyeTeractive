"""Rotula a direção do olhar ao longo de um vídeo, em uma passada.

Item que faltava para avaliar o sistema honestamente: só ``up.mp4`` tinha
rótulo conhecido, o que limitava a validação a um eixo e a uma direção.

O funcionamento é de "etiqueta ativa": durante a reprodução, cada quadro
exibido recebe o rótulo que estiver ativo no momento. Trocar de rótulo é
apertar uma tecla; o resto é assistir ao vídeo. Ao final, os quadros viram
segmentos por codificação de corrida.

Teclas::

    a s d      esquerda / centro / direita
    w x        cima / baixo
    q e        cima-esquerda / cima-direita
    z c        baixo-esquerda / baixo-direita
    n          nenhum rótulo (trecho descartado da avaliação)
    espaço     pausa
    , .        um quadro para trás / para a frente (pausado)
    - =        mais devagar / mais rápido
    u          desfaz o rótulo do quadro atual
    s maiúsculo grava e sai;  ESC sai sem gravar

Uso::

    python scripts/rotular_video.py data/videos/video1.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyeteractive.fusion import Direcao

# Os valores saem de Direcao para que a ferramenta não consiga gravar um
# rótulo que o avaliador depois não reconheça.
TECLAS = {
    ord("a"): Direcao.ESQUERDA.value,
    ord("s"): Direcao.CENTRO.value,
    ord("d"): Direcao.DIREITA.value,
    ord("w"): Direcao.CIMA.value,
    ord("x"): Direcao.BAIXO.value,
    ord("q"): Direcao.CIMA_ESQUERDA.value,
    ord("e"): Direcao.CIMA_DIREITA.value,
    ord("z"): Direcao.BAIXO_ESQUERDA.value,
    ord("c"): Direcao.BAIXO_DIREITA.value,
    ord("n"): None,
}

CORES = {
    "centro": (200, 200, 200),
    "cima": (0, 220, 0),
    "baixo": (0, 140, 255),
    "esquerda": (255, 160, 0),
    "direita": (255, 0, 200),
    "cima-esquerda": (0, 220, 220),
    "cima-direita": (120, 220, 0),
    "baixo-esquerda": (200, 100, 255),
    "baixo-direita": (0, 80, 255),
}


def para_segmentos(rotulos: dict[int, str]) -> list[dict]:
    """Codificação de corrida dos rótulos por quadro."""
    if not rotulos:
        return []
    segmentos = []
    indices = sorted(rotulos)
    inicio = anterior = indices[0]
    atual = rotulos[inicio]

    for i in indices[1:]:
        if i == anterior + 1 and rotulos[i] == atual:
            anterior = i
            continue
        segmentos.append({"inicio": inicio, "fim": anterior, "direcao": atual})
        inicio = anterior = i
        atual = rotulos[i]

    segmentos.append({"inicio": inicio, "fim": anterior, "direcao": atual})
    return segmentos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--saida", type=Path, default=None)
    args = parser.parse_args()

    destino = args.saida or Path("data/rotulos") / f"{args.video.stem}.json"
    if destino.is_file():
        print(f"aviso: {destino} já existe e será sobrescrito ao gravar")

    captura = cv2.VideoCapture(str(args.video))
    if not captura.isOpened():
        print(f"erro: não abriu {args.video}", file=sys.stderr)
        return 1

    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0

    rotulos: dict[int, str] = {}
    ativo: str | None = None
    pausado = True
    atraso = max(1, int(1000 / fps))
    indice = 0
    janela = f"rotulando {args.video.name}"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    print(__doc__.split("Teclas::")[1].split("Uso::")[0])

    posicao_lida = -1  # índice do último quadro decodificado

    while True:
        # Buscar por índice é caro; na reprodução sequencial basta ler o
        # próximo quadro. A busca fica reservada aos saltos.
        if indice != posicao_lida + 1:
            captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, quadro = captura.read()
        if not ok:
            indice = max(0, indice - 1)
            pausado = True
            captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
            ok, quadro = captura.read()
            if not ok:
                break
        posicao_lida = indice

        altura, largura = quadro.shape[:2]
        escala = min(1.0, 720 / max(altura, 1))
        tela = cv2.resize(quadro, (int(largura * escala), int(altura * escala)))
        h, w = tela.shape[:2]

        cor = CORES.get(ativo, (120, 120, 120))
        cv2.rectangle(tela, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(tela, f"{ativo or 'sem rotulo'}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.putText(tela, f"{indice}/{total}  {'PAUSA' if pausado else 'PLAY'}  "
                          f"{len(rotulos)} quadros rotulados",
                    (w - 330, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Linha do tempo com os rótulos já atribuídos.
        base = h - 16
        cv2.rectangle(tela, (0, base), (w, h), (25, 25, 25), -1)
        if total > 0:
            for i, rotulo in rotulos.items():
                x = int(i / total * w)
                cv2.line(tela, (x, base + 2), (x, h - 2), CORES.get(rotulo, (120, 120, 120)), 1)
            cursor = int(indice / total * w)
            cv2.line(tela, (cursor, base), (cursor, h), (255, 255, 255), 1)

        cv2.imshow(janela, tela)
        tecla = cv2.waitKey(0 if pausado else atraso) & 0xFF

        if tecla == 27:  # ESC
            print("saindo sem gravar")
            break
        if tecla == ord("S"):
            segmentos = para_segmentos(rotulos)
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(
                json.dumps(
                    {
                        "video": args.video.name,
                        "fps": fps,
                        "total_frames": total,
                        "quadros_rotulados": len(rotulos),
                        "segmentos": segmentos,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\ngravado: {destino}  ({len(rotulos)} quadros, {len(segmentos)} segmentos)")
            break
        if tecla == ord(" "):
            pausado = not pausado
            continue
        if tecla == ord("u"):
            rotulos.pop(indice, None)
            continue
        if tecla == ord(","):
            indice = max(0, indice - 1)
            pausado = True
            continue
        if tecla == ord("."):
            indice = min(total - 1, indice + 1)
            pausado = True
            continue
        if tecla == ord("-"):
            atraso = min(500, int(atraso * 1.5) + 1)
            continue
        if tecla == ord("="):
            atraso = max(1, int(atraso / 1.5))
            continue
        if tecla in TECLAS:
            ativo = TECLAS[tecla]
            continue

        if ativo is not None:
            rotulos[indice] = ativo
        if not pausado:
            indice += 1

    captura.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
