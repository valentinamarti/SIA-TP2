import pygame
import numpy as np

def load_image(path: str, size: tuple[int, int] = None) -> np.ndarray:
    img = pygame.image.load(path)
    if size is not None:
        img = pygame.transform.scale(img, size)
    arr = pygame.surfarray.array3d(img)
    arr = np.transpose(arr, (1, 0, 2))   # pasar a (alto, ancho, 3)
    return arr.astype(np.uint8)
