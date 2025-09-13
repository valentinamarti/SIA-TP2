def replace_population_young_bias(current_population, offspring, fitness, n_selected):
    """
        Generate the next generation with young bias:

        - If k > n_selected: select n_selected individuals from offspring only.
        - If k <= n_selected: take all offspring + (n_selected - k) best from current_population.

        Args:
            current_population (list[Individual]): current generation
            offspring (list[Individual]): new individuals generated via crossovers/mutation
            fitness (function): function to evaluate fitness of an individual
            n_selected (int): number of individuals to keep for next generation

        Returns:
            list[Individual]: next generation of size n_selected
        """
    k = len(offspring)
    sorted_offspring = sorted(offspring, key=fitness, reverse=True)

    if k >= n_selected:
        # More children than wanted individuals
        next_generation = sorted_offspring[:n_selected]
    else:
        n_remaining = n_selected - k
        sorted_current = sorted(current_population, key=fitness, reverse=True)
        next_generation = sorted_offspring + sorted_current[:n_remaining]

    return next_generation