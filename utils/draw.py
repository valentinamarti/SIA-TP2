from typing import List, Dict

import numpy as np
import pygame
import cv2
import os

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

        # Convertir BGRA -> RGBA para pygame
        b, g, r, a = poly.color
        # Pre-multiplicar RGB por alpha para BLEND_PREMULTIPLIED
        r = r * a // 255
        g = g * a // 255
        b = b * a // 255
        color = (r, g, b, a)

        # Dibujar en surface temporal para blending
        poly_surface = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.polygon(poly_surface, color, points)
        surface.blit(poly_surface, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)

    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    return rgb


def save_rendered(
    individual: Individual,
    size: tuple[int, int],
    filename: str = "results/output.png",
    *,
    background_rgba: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> None:
    """Guarda el render del individuo en disco.

    - `with_alpha=False` (default): guarda imagen de 3 canales (BGR). Sin transparencia.
    - `with_alpha=True`: guarda PNG de 4 canales (BGRA) preservando alpha.
    - `background_rgba`: color de fondo cuando `with_alpha=True`. Usar A=0 para fondo transparente.
    """
    rgb = render_individual(individual, size)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGRA))
    print(f"Saved {filename}")

