import random
from utils.polygon import Individual, create_random_individual

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
