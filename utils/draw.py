from typing import List, Dict

import numpy as np
import pygame
import cv2

from utils.polygon import Individual, PolygonGene

def render_individual(individual: Individual, size: tuple[int, int]) -> np.ndarray:
    """Renderiza y devuelve RGB (sin alpha), compuesto sobre fondo blanco.

    Se respeta la transparencia de cada polígono al dibujar, pero el resultado
    final ya está mezclado sobre blanco y se retorna en RGB.
    """
    surface = pygame.Surface(size, pygame.SRCALPHA)
    surface.fill((255, 255, 255, 255))  # fondo blanco
    w, h = size
    for poly in individual.polygons:
        # Convertir coordenadas normalizadas a píxeles
        points = [(int(x * w), int(y * h)) for x, y in poly.vertices]
        color = poly.color  # BGRA (pygame usa A para blending)
        pygame.draw.polygon(surface, color, points)
    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    return rgb


def render_individual_rgba(
    individual: Individual,
    size: tuple[int, int],
    background_rgba: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> np.ndarray:
    """Renderiza y devuelve RGBA preservando transparencia.

    - `background_rgba`: color de fondo. Usar alpha=0 para fondo transparente.
    - Devuelve un arreglo (H, W, 4) en RGBA.
    """
    surface = pygame.Surface(size, pygame.SRCALPHA)
    surface.fill(background_rgba)
    w, h = size
    for poly in individual.polygons:
        points = [(int(x * w), int(y * h)) for x, y in poly.vertices]
        color = poly.color  # BGRA
        pygame.draw.polygon(surface, color, points)

    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    alpha = pygame.surfarray.array_alpha(surface).transpose(1, 0)
    rgba = np.dstack([rgb, alpha])
    return rgba


def save_rendered(
    individual: Individual,
    size: tuple[int, int],
    filename: str = "results/output.png",
    *,
    with_alpha: bool = False,
    background_rgba: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> None:
    """Guarda el render del individuo en disco.

    - `with_alpha=False` (default): guarda imagen de 3 canales (BGR). Sin transparencia.
    - `with_alpha=True`: guarda PNG de 4 canales (BGRA) preservando alpha.
    - `background_rgba`: color de fondo cuando `with_alpha=True`. Usar A=0 para fondo transparente.
    """
    if with_alpha:
        rgba = render_individual_rgba(individual, size, background_rgba)
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(filename, bgra)
    else:
        rgb = render_individual(individual, size)
        cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved {filename}")

