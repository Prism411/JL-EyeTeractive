"""Classificador ResNet-101 do olhar e marginalização por eixo.

O modelo treinado prevê cinco classes mutuamente exclusivas — ``center``,
``down``, ``left``, ``right``, ``up`` — o que à primeira vista não cobre as
diagonais. Cobre: cada classe é, na verdade, um **par (horizontal, vertical)**
em que um dos eixos está no centro::

    center → (centro,   centro)
    left   → (esquerda, centro)
    right  → (direita,  centro)
    up     → (centro,   cima)
    down   → (centro,   baixo)

Marginalizando o softmax sobre cada eixo obtém-se evidência independente para
horizontal e vertical::

    P(vertical = cima)   = p_up
    P(vertical = baixo)  = p_down
    P(vertical = centro) = p_center + p_left + p_right

    P(horizontal = esquerda) = p_left
    P(horizontal = direita)  = p_right
    P(horizontal = centro)   = p_center + p_up + p_down

Um olhar diagonal reparte a massa entre, digamos, ``up`` e ``left``; nas
marginais isso aparece como evidência simultânea de "cima" e de "esquerda",
que a camada paraconsistente combina numa das nove direções. É por isso que a
CNN pode assumir o eixo vertical sem precisar de classes novas nem de retreino.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .paraconsistent import Anotacao, evidencia_de_probabilidades

__all__ = ["CLASSES", "MarginaisCNN", "marginalizar", "ClassificadorOlhar"]

# Ordem alfabética — é a que ``torchvision.datasets.ImageFolder`` produz e a
# que foi usada no treino. Alterar esta lista invalida os pesos salvos.
CLASSES: tuple[str, ...] = ("center", "down", "left", "right", "up")

_IDX = {nome: i for i, nome in enumerate(CLASSES)}

# Normalização do ImageNet, idêntica à usada no treino.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)
TAMANHO_ENTRADA = (224, 224)


@dataclass(frozen=True)
class MarginaisCNN:
    """Marginais por eixo extraídas do softmax de cinco classes."""

    cima: float
    baixo: float
    centro_vertical: float
    esquerda: float
    direita: float
    centro_horizontal: float
    probabilidades: tuple[float, ...]

    @property
    def classe_argmax(self) -> str:
        return CLASSES[int(np.argmax(self.probabilidades))]

    def anotacao_vertical_cima(self) -> Anotacao:
        """Anotação para "o olhar está para CIMA".

        A massa de ``centro_vertical`` fica deliberadamente de fora: ela não
        apoia nem refuta a proposição, e mantê-la fora empurra a anotação para
        a região paracompleta quando o olhar está de fato no centro.
        """
        return evidencia_de_probabilidades(self.cima, self.baixo)

    def anotacao_vertical_baixo(self) -> Anotacao:
        """Anotação para "o olhar está para BAIXO"."""
        return evidencia_de_probabilidades(self.baixo, self.cima)

    def anotacao_horizontal_direita(self) -> Anotacao:
        """Anotação para "o olhar está para a DIREITA"."""
        return evidencia_de_probabilidades(self.direita, self.esquerda)


def marginalizar(probabilidades: Sequence[float]) -> MarginaisCNN:
    """Projeta o softmax de cinco classes nas marginais de cada eixo."""
    if len(probabilidades) != len(CLASSES):
        raise ValueError(f"Esperadas {len(CLASSES)} probabilidades, recebidas {len(probabilidades)}.")

    p = [float(x) for x in probabilidades]
    p_center = p[_IDX["center"]]
    p_down = p[_IDX["down"]]
    p_left = p[_IDX["left"]]
    p_right = p[_IDX["right"]]
    p_up = p[_IDX["up"]]

    return MarginaisCNN(
        cima=p_up,
        baixo=p_down,
        centro_vertical=p_center + p_left + p_right,
        esquerda=p_left,
        direita=p_right,
        centro_horizontal=p_center + p_up + p_down,
        probabilidades=tuple(p),
    )


class ClassificadorOlhar:
    """Carrega a ResNet-101 treinada e infere marginais a partir de recortes."""

    def __init__(
        self,
        pesos: str | Path,
        dispositivo: str | None = None,
        inverter_horizontal: bool = False,
        escala_de_cinza: bool = True,
        limite_memoria_gpu: float | None = None,
    ) -> None:
        """Carrega o classificador.

        ``escala_de_cinza`` converte o recorte para cinza antes de replicá-lo
        em três canais. O padrão é ``True`` porque é assim que o modelo foi
        treinado: ``ai_train/data_extractor.py`` recortava de ``gray``, então
        todo o dataset é ``mode='L'``. Alimentar a rede com cor verdadeira é um
        descasamento de distribuição.

        ``limite_memoria_gpu`` é a fração da VRAM que este processo pode
        reservar (por exemplo ``0.25``). Serve para conviver com outros
        trabalhos na mesma placa.
        """
        import torch
        from torchvision import models

        self._torch = torch
        self.dispositivo = torch.device(
            dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.inverter_horizontal = inverter_horizontal
        self.escala_de_cinza = escala_de_cinza

        if limite_memoria_gpu is not None and self.dispositivo.type == "cuda":
            if not 0.0 < limite_memoria_gpu <= 1.0:
                raise ValueError("limite_memoria_gpu deve estar em (0, 1].")
            torch.cuda.set_per_process_memory_fraction(
                limite_memoria_gpu, self.dispositivo.index or 0
            )

        caminho = Path(pesos)
        if not caminho.is_file():
            raise FileNotFoundError(f"Pesos do modelo não encontrados: {caminho}")

        modelo = models.resnet101(weights=None)
        modelo.fc = torch.nn.Linear(modelo.fc.in_features, len(CLASSES))
        estado = torch.load(caminho, map_location=self.dispositivo)
        modelo.load_state_dict(estado)
        self.modelo = modelo.to(self.dispositivo).eval()

        self._media = torch.tensor(MEDIA_IMAGENET, device=self.dispositivo).view(3, 1, 1)
        self._desvio = torch.tensor(DESVIO_IMAGENET, device=self.dispositivo).view(3, 1, 1)

    def _preparar(self, recorte_rgb: np.ndarray):
        """Converte um recorte RGB ``HxWx3`` uint8 no tensor esperado pela rede."""
        import cv2

        torch = self._torch
        if recorte_rgb.shape[:2] != TAMANHO_ENTRADA:
            recorte_rgb = cv2.resize(recorte_rgb, TAMANHO_ENTRADA, interpolation=cv2.INTER_AREA)

        if self.escala_de_cinza:
            cinza = cv2.cvtColor(recorte_rgb, cv2.COLOR_RGB2GRAY)
            recorte_rgb = cv2.cvtColor(cinza, cv2.COLOR_GRAY2RGB)

        tensor = torch.from_numpy(np.ascontiguousarray(recorte_rgb)).to(self.dispositivo)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - self._media) / self._desvio
        return tensor.unsqueeze(0)

    def prever(self, recorte_rgb: np.ndarray) -> MarginaisCNN:
        """Infere as marginais por eixo para um recorte do olho."""
        torch = self._torch
        entrada = self._preparar(recorte_rgb)
        with torch.inference_mode():
            logits = self.modelo(entrada)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        marginais = marginalizar(probs)
        if self.inverter_horizontal:
            marginais = MarginaisCNN(
                cima=marginais.cima,
                baixo=marginais.baixo,
                centro_vertical=marginais.centro_vertical,
                esquerda=marginais.direita,
                direita=marginais.esquerda,
                centro_horizontal=marginais.centro_horizontal,
                probabilidades=marginais.probabilidades,
            )
        return marginais

    def prever_lote(self, recortes_rgb: Sequence[np.ndarray]) -> list[MarginaisCNN]:
        """Versão em lote de :meth:`prever`, para avaliação offline."""
        torch = self._torch
        if not recortes_rgb:
            return []
        lote = torch.cat([self._preparar(r) for r in recortes_rgb], dim=0)
        with torch.inference_mode():
            probs = torch.softmax(self.modelo(lote), dim=1).cpu().numpy()

        saidas = []
        for linha in probs:
            marginais = marginalizar(linha)
            if self.inverter_horizontal:
                marginais = MarginaisCNN(
                    cima=marginais.cima,
                    baixo=marginais.baixo,
                    centro_vertical=marginais.centro_vertical,
                    esquerda=marginais.direita,
                    direita=marginais.esquerda,
                    centro_horizontal=marginais.centro_horizontal,
                    probabilidades=marginais.probabilidades,
                )
            saidas.append(marginais)
        return saidas
