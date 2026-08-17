#!/usr/bin/env bash
# Baixa os ativos de modelo que não são versionados no repositório.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINO="$RAIZ/models"
mkdir -p "$DESTINO"

LANDMARKER_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if [[ -f "$DESTINO/face_landmarker.task" ]]; then
  echo "face_landmarker.task já presente, pulando."
else
  echo "baixando face_landmarker.task ..."
  curl -fsSL -o "$DESTINO/face_landmarker.task" "$LANDMARKER_URL"
  echo "ok: $DESTINO/face_landmarker.task"
fi

if [[ ! -f "$DESTINO/resnet101_model.pth" ]]; then
  cat >&2 <<'AVISO'

ATENÇÃO: models/resnet101_model.pth não está presente.

Os pesos treinados não são distribuídos publicamente porque derivam de vídeos
de voluntários. Obtenha-os com a equipe do projeto ou treine do zero:

    python scripts/treinar.py --dataset data/dataset --epocas 35 --amp

AVISO
fi
