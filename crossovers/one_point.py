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