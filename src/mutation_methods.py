from utils.polygon import Individual, PolygonGene
from typing import List
from utils.polygon import create_random_polygon_gene
import random
import numpy as np

MAX_POLYGONS = 100

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

def add_polygon(ind, prob, max_polygons):
        """
        Mutación estructural: puede agregar polígonos al individuo.
        Mantiene la longitud del genoma menor o igual a max_polygons
        """
        new_ind = ind.copy()

        if len(new_ind.polygons) < max_polygons:
            new_polygon = create_random_polygon_gene(ind.polygons[0].n_vertices) 
            new_ind.polygons.append(new_polygon)
                
        return new_ind


def mutate_individual(mutation_method: str, ind: Individual, size: tuple[int, int], prob=0.1, structural: bool = False) -> Individual:
    def gen(ind: Individual, prob: float) -> Individual:
        new_ind = ind.copy()
        if random.random() < prob:
            idx = random.randrange(len(new_ind.polygons))
            mutated = mutate_gen(new_ind.polygons[idx], prob)
            new_ind.polygons[idx] = mutated

        if structural and random.random() < prob:
            new_ind = add_polygon(new_ind, prob, MAX_POLYGONS)
        return new_ind

    def limited_multi_gen(ind, prob, structural: bool = False):
        """
        Limited multi gen mutation: muta un número limitado de polígonos (máximo 10).
        Selecciona aleatoriamente qué polígonos mutar y los muta con la probabilidad dada.
        """
        new_ind = ind.copy()
        
        max_mutations = min(10, len(new_ind.polygons))
        num_mutations = random.randint(1, max_mutations)
        
        polygons_to_mutate = random.sample(range(len(new_ind.polygons)), num_mutations)
        
        for i in polygons_to_mutate:
            if random.random() < prob:
                new_ind.polygons[i] = mutate_gen(new_ind.polygons[i], prob)
        
        if structural and random.random() < prob:
            new_ind = add_polygon(new_ind, prob, MAX_POLYGONS)

        return new_ind

    def uniform_multi_gen(ind, prob, structural: bool = False):
        """
        Uniform multi gen mutation: cada polígono tiene probabilidad independiente 
        de ser mutado, sin límite en el número de mutaciones.
        """
        new_ind = ind.copy()
        
        for i in range(len(new_ind.polygons)):
            if random.random() < prob:
                # Mutar este polígono específico
                new_ind.polygons[i] = mutate_gen(new_ind.polygons[i], prob)

        if structural and random.random() < prob:
            new_ind = add_polygon(new_ind, prob, MAX_POLYGONS)
        
        return new_ind

    def complete(ind, prob, structural: bool = False):
        
        return ind.copy()

    methods = {
        "gen": gen,
        "limited_multi_gen": limited_multi_gen,
        "uniform_multi_gen": uniform_multi_gen,
        "complete": complete,
    }
    return methods[mutation_method](ind, prob)
