import random

from utils.polygon import Individual


def one_point_crossover(individual_1, individual_2):
    """
        Perform one point crossovers between two individuals.

        Args:
            individual_1 (Individual)
            individual_2 (Individual)
        Returns:
            tuple: two offspring individuals
    """

    s = len(individual_1.polygons)

    locus = random.randint(0, s - 1)

    child_1_genes = (
            individual_1.polygons[:locus] + individual_2.polygons[locus:]
    )

    child_2_genes = (
            individual_2.polygons[:locus] + individual_1.polygons[locus:]
    )

    return Individual(child_1_genes), Individual(child_2_genes)


def two_point_crossover(individual_1, individual_2):
    """
        Perform two point crossovers between two individuals.

        Args:
            individual_1 (Individual)
            individual_2 (Individual)
        Returns:
            tuple: two offspring individuals
    """
    s = len(individual_1.polygons)

    # We choose to random points, and we sort them so that p_1 <= p_2
    locus_1, locus_2 = sorted([random.randint(0, s - 1), random.randint(0, s - 1)])

    child_1_genes = (
        individual_1.polygons[:locus_1] + individual_2.polygons[locus_1:locus_2] + individual_1.polygons[locus_2:]
    )

    child_2_genes = (
            individual_2.polygons[:locus_1] + individual_1.polygons[locus_1:locus_2] + individual_2.polygons[locus_2:]
    )

    return Individual(child_1_genes), Individual(child_2_genes)




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