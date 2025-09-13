from random import random, seed
from utils.polygon import Individual


def uniform_crossover(individual_1, individual_2, p=0.5):
    """
        Perform uniform crossovers between two individuals.

        Args:
            individual_1 (Individual)
            individual_2 (Individual)
            p (float): probability of taking gene from individual_1
        Returns:
            tuple: two offspring individuals
    """
    child_1_genes = []
    child_2_genes = []

    for allele1, allele2 in zip(individual_1.polygons, individual_2.polygons):
        if random.random() < p:
            child_1_genes.append(allele1)
            child_2_genes.append(allele2)
        else:
            child_1_genes.append(allele2)
            child_2_genes.append(allele1)

    return Individual(child_1_genes), Individual(child_2_genes)