"""Visualização ao vivo: webcam com a direção e as evidências sobrepostas.

Mostra os dois eixos separadamente, com grau de certeza real e estado lógico,
para tornar visível *qual fonte* está decidindo cada eixo — a leitura que o
protótipo não oferecia.

Uso::

    python scripts/webcam.py --camera 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyeteractive.pipeline import PipelineOlhar

VERDE = (0, 200, 0)
AMBAR = (0, 165, 255)
VERMELHO = (0, 0, 220)
BRANCO = (245, 245, 245)

CORES_ESTADO = {
    "verdadeiro": VERDE,
    "falso": VERDE,
    "indeterminado": AMBAR,
    "paracompleto": AMBAR,
    "inconsistente": VERMELHO,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="índice da webcam ou caminho de vídeo")
    parser.add_argument("--pesos", type=Path, default=Path("models/resnet101_model.pth"))
    parser.add_argument("--landmarker", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--espelhar", action="store_true", help="inverte a imagem para exibição")
    args = parser.parse_args()

    fonte: int | str = int(args.camera) if args.camera.isdigit() else args.camera
    captura = cv2.VideoCapture(fonte)
    if not captura.isOpened():
        print(f"erro: não abriu a fonte {args.camera}", file=sys.stderr)
        return 1

    inicio = time.monotonic()
    ultimo = inicio
    fps = 0.0

    with PipelineOlhar(pesos_cnn=args.pesos, modelo_landmarker=args.landmarker) as pipeline:
        while True:
            ok, quadro = captura.read()
            if not ok:
                break

            agora = time.monotonic()
            analise = pipeline.processar(quadro, timestamp_ms=int((agora - inicio) * 1000))
            fps = 0.9 * fps + 0.1 / max(agora - ultimo, 1e-6)
            ultimo = agora

            exibicao = cv2.flip(quadro, 1) if args.espelhar else quadro.copy()
            altura, largura = exibicao.shape[:2]

            if analise.rosto_detectado:
                resultado = analise.resultado
                if analise.caixa_olho and not args.espelhar:
                    x, y, w, h = analise.caixa_olho
                    cv2.rectangle(exibicao, (x, y), (x + w, y + h), VERDE, 1)

                cv2.putText(
                    exibicao,
                    resultado.direcao.value.upper(),
                    (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    VERDE if resultado.estavel else AMBAR,
                    2,
                )

                linhas = [
                    (
                        f"V {resultado.vertical.valor:<8} Gcr={resultado.vertical.certeza_real:+.2f} "
                        f"{resultado.vertical.estado.value}",
                        CORES_ESTADO[resultado.vertical.estado.value],
                    ),
                    (
                        f"H {resultado.horizontal.valor:<8} Gcr={resultado.horizontal.certeza_real:+.2f} "
                        f"{resultado.horizontal.estado.value}",
                        CORES_ESTADO[resultado.horizontal.estado.value],
                    ),
                ]
                if analise.medidas is not None:
                    linhas.append(
                        (
                            f"geometria  h={analise.medidas.grau_horizontal:+6.2f}  "
                            f"v={analise.medidas.grau_vertical:+5.2f}  "
                            f"EAR={analise.medidas.abertura:.2f}",
                            BRANCO,
                        )
                    )
                if analise.marginais is not None:
                    m = analise.marginais
                    linhas.append(
                        (
                            f"CNN  cima={m.cima:.2f} baixo={m.baixo:.2f}  "
                            f"esq={m.esquerda:.2f} dir={m.direita:.2f}",
                            BRANCO,
                        )
                    )

                for i, (texto, cor) in enumerate(linhas):
                    cv2.putText(
                        exibicao, texto, (12, 72 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 1,
                    )
            else:
                cv2.putText(
                    exibicao, "sem rosto", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, VERMELHO, 2,
                )

            cv2.putText(
                exibicao, f"{fps:4.1f} fps", (largura - 110, altura - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BRANCO, 1,
            )
            cv2.imshow("EyeTeractive", exibicao)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    captura.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
