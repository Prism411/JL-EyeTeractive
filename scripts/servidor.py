"""Sobe o servidor de inferência do olhar.

Uso::

    python scripts/servidor.py --porta 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyeteractive.io.protocolo import PORTA_PADRAO
from eyeteractive.io.servidor import ServidorOlhar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pesos", type=Path, default=Path("models/resnet101_model.pth"))
    parser.add_argument("--landmarker", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--endereco", default="0.0.0.0")
    parser.add_argument("--porta", type=int, default=PORTA_PADRAO)
    parser.add_argument("--pular-quadros", type=int, default=1)
    parser.add_argument(
        "--apenas-cardeais",
        action="store_true",
        help="projeta as diagonais nos códigos 1..4 para clientes antigos",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
    )

    servidor = ServidorOlhar(
        pesos_cnn=args.pesos,
        modelo_landmarker=args.landmarker,
        endereco=args.endereco,
        porta=args.porta,
        pular_quadros=args.pular_quadros,
        apenas_cardeais=args.apenas_cardeais,
    )
    try:
        servidor.servir()
    except KeyboardInterrupt:
        logging.info("encerrando a pedido do usuário")
    finally:
        servidor.fechar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
