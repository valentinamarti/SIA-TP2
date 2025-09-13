from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from utils.draw import render_individual
from utils.polygon import Individual


def _mse_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized MSE in [0, 1] over RGB images in uint8.

    0 means identical; 1 means maximal average squared error (255^2).
    """
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    err = np.mean((a32 - b32) ** 2) / (255.0 ** 2)
    return float(err)


def _edges(gray: np.ndarray) -> np.ndarray:
    """Return Canny edges normalized to [0,1]. Input gray uint8."""
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return (edges.astype(np.float32) / 255.0)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


class FitnessEvaluator:

    def __init__(
        self,
        target_rgb: np.ndarray,
        size_wh: tuple[int, int] | None = None,
        *,
        w_color: float = 0.7,
        w_edge: float = 0.3,
    ) -> None:
        if size_wh is None:
            size_wh = (target_rgb.shape[1], target_rgb.shape[0])
        self.size_wh = size_wh

        # Store target as uint8 RGB
        self.target_rgb: np.ndarray = target_rgb.astype(np.uint8)
        self._target_gray: np.ndarray | None = None
        self._target_edges: np.ndarray | None = None
        self._target_pyr: list[np.ndarray] | None = None

        # Defaults (can be overridden per call)
        self.w_color_default = w_color
        self.w_edge_default = w_edge

        # Lazy compute caches
        self._ensure_target_gray_edges()

    def _ensure_target_gray_edges(self) -> None:
        if self._target_gray is None:
            self._target_gray = _to_gray(self.target_rgb)
        if self._target_edges is None:
            self._target_edges = _edges(self._target_gray)


    def evaluate_individual(
        self,
        individual: Individual,
        *,
        w_color: float | None = None,
        w_edge: float | None = None,
    ) -> Individual:
        # Defaults
        if w_color is None:
            w_color = self.w_color_default
        if w_edge is None:
            w_edge = self.w_edge_default

        # Render candidate
        cand = render_individual(individual, self.size_wh)

        # Ensure caches
        self._ensure_target_gray_edges()

        # Color error
        color_err = _mse_norm(self.target_rgb, cand)

        # Edge error (single scale)
        cand_gray = _to_gray(cand)
        cand_edges = _edges(cand_gray)
        edge_err = float(np.mean((self._target_edges - cand_edges) ** 2))

        # Combine
        w_sum = max(1e-8, (w_color + w_edge))
        color_w = w_color / w_sum
        edge_w = w_edge / w_sum
        error = color_w * color_err + edge_w * edge_err

        fitness = float(np.clip(1.0 - error, 0.0, 1.0))
        individual.fitness = fitness
        individual.error = error
        return individual

    def evaluate_population(
        self,
        population: Iterable[Individual],
        *,
        w_color: float | None = None,
        w_edge: float | None = None,
    ) -> None:
        for ind in population:
            self.evaluate_individual(
                ind,
                w_color=w_color,
                w_edge=w_edge,
            )
