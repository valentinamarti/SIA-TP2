from utils.polygon import Individual, PolygonGene
import random
from typing import List
import numpy as np


def mutate_gen(polygon: PolygonGene, prob: float) -> PolygonGene:
    new_polygon = polygon.copy()
    # mover un vértice
    if random.random() < prob:
        i = random.randrange(new_polygon.n_vertices)
        x, y = new_polygon.vertices[i]
        dx, dy = random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)
        new_x = min(max(x + dx, 0.0), 1.0)
        new_y = min(max(y + dy, 0.0), 1.0)
        new_polygon.vertices[i] = (new_x, new_y)
    # mutar color
    if random.random() < prob:
        b, g, r, a = new_polygon.color
        b = int(np.clip(b + random.randint(-10, 10), 0, 255))
        g = int(np.clip(g + random.randint(-10, 10), 0, 255))
        r = int(np.clip(r + random.randint(-10, 10), 0, 255))
        a = int(np.clip(a + random.randint(-10, 10), 0, 255))
        new_polygon.color = (b, g, r, a)
    return new_polygon


def mutate_individual(mutation_method: str, ind: Individual, size: tuple[int, int], prob=0.1) -> Individual:
    def gen(ind: Individual, prob: float) -> Individual:
        new_ind = ind.copy()
        if random.random() < prob:
            idx = random.randrange(len(new_ind.polygons))
            mutated = mutate_gen(new_ind.polygons[idx], prob)
            new_ind.polygons[idx] = mutated
        return new_ind

    def limited_multi_gen(ind, prob):
        return ind.copy()

    def uniform_multi_gen(ind, prob):
        return ind.copy()

    def complete(ind, prob):
        return ind.copy()

    methods = {
        "gen": gen,
        "limited_multi_gen": limited_multi_gen,
        "uniform_multi_gen": uniform_multi_gen,
        "complete": complete,
    }
    return methods[mutation_method](ind, prob)
