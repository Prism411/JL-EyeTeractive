"""Calibra os parâmetros de fusão para um usuário e grava o perfil.

Três modos de coleta:

``--fonte webcam``
    Sessão interativa: um alvo aparece em cinco posições da tela e os quadros
    são coletados enquanto o usuário o fixa. É o modo de uso real.

``--fonte rotulos``
    Offline, a partir de vídeos rotulados quadro a quadro (``data/rotulos``).
    Enquadramento de gravação real, e ``--apenas-videos`` permite calibrar num
    conjunto de sessões e avaliar em outro.

``--fonte frames``
    Offline, a partir de um diretório de frames rotulados por classe
    (``center/``, ``down/``, ``left/``, ``right/``, ``up/``).

O perfil resultante entra no pipeline por
``ConfigFusao`` via ``eyeteractive.calibracao.carregar_perfil``.

Uso::

    python scripts/calibrar.py --fonte webcam --usuario jader
    python scripts/calibrar.py --fonte rotulos --apenas-videos video_9.mp4 video_13.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyeteractive.calibracao import (
    ALVOS,
    AmostraCalibracao,
    calibrar,
    salvar_perfil,
)
from eyeteractive.cnn import CLASSES, ClassificadorOlhar
from eyeteractive.fusion import Direcao
from eyeteractive.geometry import medir
from eyeteractive.landmarks import DetectorFacial

# Onde desenhar o alvo, em fração da tela, para cada linha de ALVOS.
POSICAO_ALVO = {
    "centro": (0.50, 0.50),
    "cima": (0.50, 0.10),
    "baixo": (0.50, 0.90),
    "esquerda": (0.08, 0.50),
    "direita": (0.92, 0.50),
}

VERTICAL_ESPERADO = {"up": "cima", "down": "baixo", "center": "centro", "left": "centro", "right": "centro"}
HORIZONTAL_ESPERADO = {"left": "esquerda", "right": "direita", "center": "centro", "up": "centro", "down": "centro"}


def coletar_de_rotulos(
    pasta_rotulos: Path,
    pastas_video: list[Path],
    detector: DetectorFacial,
    classificador: ClassificadorOlhar,
    apenas: set[str] | None = None,
) -> list[AmostraCalibracao]:
    """Coleta a partir de vídeos rotulados por quadro.

    É a fonte mais fiel das três: o enquadramento é o da gravação real, e não
    o de uma sessão de calibração encenada. ``apenas`` restringe a um
    subconjunto de vídeos — necessário para calibrar num conjunto e avaliar em
    outro, sem calibrar sobre o que se pretende medir.
    """
    amostras: list[AmostraCalibracao] = []

    for arquivo in sorted(pasta_rotulos.glob("*.json")):
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        nome = dados["video"]
        if apenas is not None and nome not in apenas:
            continue

        caminho = next((p / nome for p in pastas_video if (p / nome).is_file()), None)
        if caminho is None:
            continue

        por_quadro: dict[int, str] = {}
        for segmento in dados["segmentos"]:
            for i in range(segmento["inicio"], segmento["fim"] + 1):
                por_quadro[i] = segmento["direcao"]

        detector.reiniciar()
        captura = cv2.VideoCapture(str(caminho))
        fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
        indice = 0
        while True:
            ok, quadro = captura.read()
            if not ok:
                break
            direcao = por_quadro.get(indice)
            indice += 1
            if direcao is None:
                continue
            deteccao = detector.detectar(quadro, timestamp_ms=int(indice * 1000 / fps))
            if deteccao is None:
                continue
            vertical, horizontal = Direcao(direcao).eixos()
            amostras.append(
                AmostraCalibracao(
                    marginais=classificador.prever(deteccao.recorte_rgb),
                    medidas=medir(deteccao.contorno, deteccao.iris),
                    vertical_esperado=vertical.value,
                    horizontal_esperado=horizontal.value,
                )
            )
        captura.release()
        print(f"  {nome:<16} {len(amostras):>5} amostras acumuladas")

    return amostras


def coletar_de_frames(
    pasta: Path, detector: DetectorFacial, classificador: ClassificadorOlhar
) -> list[AmostraCalibracao]:
    amostras = []
    for classe in CLASSES:
        diretorio = pasta / classe
        if not diretorio.is_dir():
            continue
        for caminho in sorted(diretorio.iterdir()):
            if caminho.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            quadro = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
            if quadro is None:
                continue
            deteccao = detector.detectar(quadro)
            if deteccao is None:
                continue
            amostras.append(
                AmostraCalibracao(
                    marginais=classificador.prever(deteccao.recorte_rgb),
                    medidas=medir(deteccao.contorno, deteccao.iris),
                    vertical_esperado=VERTICAL_ESPERADO[classe],
                    horizontal_esperado=HORIZONTAL_ESPERADO[classe],
                )
            )
    return amostras


def coletar_da_webcam(
    camera: int,
    detector: DetectorFacial,
    classificador: ClassificadorOlhar,
    por_alvo: int,
    segundos_preparo: float,
) -> list[AmostraCalibracao]:
    captura = cv2.VideoCapture(camera)
    if not captura.isOpened():
        raise SystemExit(f"erro: não abriu a câmera {camera}")

    janela = "Calibração EyeTeractive"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    amostras: list[AmostraCalibracao] = []
    largura, altura = 1280, 720
    inicio_sessao = time.monotonic()

    try:
        for nome, vertical, horizontal in ALVOS:
            fx, fy = POSICAO_ALVO[nome]
            alvo = (int(fx * largura), int(fy * altura))
            coletados = 0
            comeco = time.monotonic()

            while coletados < por_alvo:
                ok, quadro = captura.read()
                if not ok:
                    break

                decorrido = time.monotonic() - comeco
                preparando = decorrido < segundos_preparo

                tela = cv2.resize(quadro, (largura, altura))
                tela = cv2.addWeighted(tela, 0.25, tela, 0.0, 0)  # escurece o fundo

                cor = (0, 180, 255) if preparando else (0, 255, 0)
                cv2.circle(tela, alvo, 26, cor, 3)
                cv2.circle(tela, alvo, 6, cor, -1)
                if preparando:
                    restante = segundos_preparo - decorrido
                    texto = f"olhe para o alvo — {nome} ({restante:.1f}s)"
                else:
                    texto = f"coletando {nome}: {coletados}/{por_alvo}"
                cv2.putText(tela, texto, (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
                cv2.putText(tela, "ESC cancela", (30, altura - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow(janela, tela)

                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt

                if preparando:
                    continue

                ts = int((time.monotonic() - inicio_sessao) * 1000)
                deteccao = detector.detectar(quadro, timestamp_ms=ts)
                if deteccao is None:
                    continue
                amostras.append(
                    AmostraCalibracao(
                        marginais=classificador.prever(deteccao.recorte_rgb),
                        medidas=medir(deteccao.contorno, deteccao.iris),
                        vertical_esperado=vertical,
                        horizontal_esperado=horizontal,
                    )
                )
                coletados += 1
            print(f"  {nome:<9} {coletados} amostras")
    except KeyboardInterrupt:
        print("\ncalibração cancelada pelo usuário")
    finally:
        captura.release()
        cv2.destroyAllWindows()

    return amostras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonte", choices=("webcam", "frames", "rotulos"), default="webcam")
    parser.add_argument("--frames", type=Path, default=Path("data/frames_rotulados"))
    parser.add_argument("--rotulos", type=Path, default=Path("data/rotulos"))
    parser.add_argument("--videos", type=Path, nargs="+",
                        default=[Path("data/videos_fonte"), Path("data/videos")])
    parser.add_argument("--apenas-videos", nargs="*", default=None,
                        help="restringe a calibração a estes vídeos")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--usuario", default="")
    parser.add_argument("--perfil", type=Path, default=None)
    parser.add_argument("--pesos", type=Path, default=Path("models/resnet101_model.pth"))
    parser.add_argument("--landmarker", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--por-alvo", type=int, default=40)
    parser.add_argument("--preparo", type=float, default=2.0)
    parser.add_argument("--limite-gpu", type=float, default=0.25)
    args = parser.parse_args()

    destino = args.perfil or Path(f"perfis/{args.usuario or 'usuario'}.json")

    classificador = ClassificadorOlhar(args.pesos, limite_memoria_gpu=args.limite_gpu)
    modo_video = args.fonte in ("webcam", "rotulos")

    with DetectorFacial(args.landmarker, modo_video=modo_video) as detector:
        if args.fonte == "webcam":
            print("coletando da webcam...")
            amostras = coletar_da_webcam(
                args.camera, detector, classificador, args.por_alvo, args.preparo
            )
        elif args.fonte == "rotulos":
            print(f"coletando de {args.rotulos}...")
            amostras = coletar_de_rotulos(
                args.rotulos,
                args.videos,
                detector,
                classificador,
                apenas=set(args.apenas_videos) if args.apenas_videos else None,
            )
        else:
            print(f"coletando de {args.frames}...")
            amostras = coletar_de_frames(args.frames, detector, classificador)

    if len(amostras) < 10:
        print(
            f"erro: só {len(amostras)} amostras utilizáveis — insuficiente para calibrar",
            file=sys.stderr,
        )
        return 1

    cobertura = {}
    for a in amostras:
        cobertura[(a.vertical_esperado, a.horizontal_esperado)] = (
            cobertura.get((a.vertical_esperado, a.horizontal_esperado), 0) + 1
        )
    print(f"\n{len(amostras)} amostras: {cobertura}\n")

    faltando = [
        alvo for alvo in {("cima", "centro"), ("baixo", "centro"),
                          ("centro", "esquerda"), ("centro", "direita"), ("centro", "centro")}
        if alvo not in cobertura
    ]
    if faltando:
        print(f"AVISO: alvos sem amostra: {faltando}")
        print("       os parâmetros do eixo afetado ficam mal determinados.\n")

    resultado = calibrar(amostras)
    print(resultado.resumo())
    print(f"\nlimiar vertical    {resultado.config.limiar_decisao_vertical:.3f}")
    print(f"escala vertical    {resultado.config.escala_vertical:.2f}")
    print(f"limiar horizontal  {resultado.config.limiar_decisao_horizontal:.3f}")
    print(f"escala horizontal  {resultado.config.escala_horizontal:.2f}")

    salvar_perfil(destino, resultado, usuario=args.usuario)
    print(f"\nperfil gravado em: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
