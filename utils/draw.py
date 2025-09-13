from typing import List, Dict

import numpy as np
import pygame
import cv2

from utils.polygon import Individual, PolygonGene

def render_individual(individual: Individual, size: tuple[int, int]) -> np.ndarray:
    surface = pygame.Surface(size, pygame.SRCALPHA)
    surface.fill((255, 255, 255, 255))  # fondo blanco
    w, h = size
    for poly in individual.polygons:
        # Convertir coordenadas normalizadas a píxeles
        points = [(int(x * w), int(y * h)) for x, y in poly.vertices]
        color = poly.color  # BGRA
        pygame.draw.polygon(surface, color, points)
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr, (1, 0, 2))
    return arr


def save_rendered(individual: Individual, size, filename="results/output.png"):
    arr = render_individual(individual, size)
    cv2.imwrite(filename, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
