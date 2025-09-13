from dataclasses import dataclass
from typing import Tuple, List, Optional
import random


@dataclass
class PolygonGene:
    n_vertices: int
    vertices: List[Tuple[float,float]]  # coords en [0,1]
    color: Tuple[int,int,int,int]       # BGRA (OpenCV usa BGR; incluimos alpha)

    def copy(self):
        return PolygonGene(self.n_vertices, [tuple(v) for v in self.vertices], tuple(self.color))

@dataclass
class Individual:
    polygons: List[PolygonGene]
    fitness: Optional[float]=None
    error: Optional[float]=None
    def copy(self): return Individual([p.copy() for p in self.polygons], self.fitness, self.error)

def create_random_polygon_gene(n_vertices: int) -> PolygonGene:
    vertices = [(random.random(), random.random()) for _ in range(n_vertices)]
    color = (
        random.randint(0, 255),  # B
        random.randint(0, 255),  # G
        random.randint(0, 255),  # R
        random.randint(50, 150)  # A
    )
    return PolygonGene(n_vertices, vertices, color)

def create_random_individual(num_polygons: int, polygon_sides: int) -> Individual:
    polygons = [create_random_polygon_gene(polygon_sides) for _ in range(num_polygons)]
    return Individual(polygons)