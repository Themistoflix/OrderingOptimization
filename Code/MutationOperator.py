import numpy as np
import copy


class MutationOperator:
    def __init__(self, max_exchanges):
        self.max_exchanges = max_exchanges
        self.probability_distribution = np.array([1. / (n ** (3. / 2.)) for n in range(1, max_exchanges + 1, 1)])
        self.probability_distribution = self.probability_distribution / np.sum(self.probability_distribution)

    def random_mutation(self, particle):
        symbol1 = particle.atoms.getSymbols()[0]
        symbol2 = particle.atoms.getSymbols()[1]

        n_exchanges = 1 + np.random.choice(self.max_exchanges, p=self.probability_distribution)

        symbol1_indices = np.random.choice(particle.atoms.getIndicesBySymbol(symbol1), n_exchanges, replace=False)
        symbol2_indices = np.random.choice(particle.atoms.getIndicesBySymbol(symbol2), n_exchanges, replace=False)

        particle.atoms.swapAtoms(zip(symbol1_indices, symbol2_indices))

        return particle, zip(symbol1_indices, symbol2_indices)

    def revert_mutation(self, particle, swaps):
        particle.atoms.swapAtoms(swaps)

        return particle

    def gradient_descent_mutation(self,  particle, energy_coefficients):
        # will only work for bimetallic nanoparticles
        def get_n_distinct_atomic_environments():
            feature_vector = particle.getFeatureVector()
            n_distinct_environments = int(len(feature_vector) / 2)

            return n_distinct_environments

        def get_feature_index_of_other_element(symbol1_feature_index):
            feature_vector = particle.getFeatureVector()
            n_distinct_environments = int(len(feature_vector) / 2)
            symbol2_feature_index = symbol1_feature_index + n_distinct_environments

            return symbol2_feature_index


        def compute_energy_gain_for_equal_env_swaps():
            n_distinct_environments = get_n_distinct_atomic_environments()

            energy_gains = list()
            for symbol1_feature_index in range(n_distinct_environments):
                symbol2_feature_index = get_feature_index_of_other_element(symbol1_feature_index)
                energy_gain_per_swap = energy_coefficients[symbol1_feature_index] - energy_coefficients[symbol2_feature_index]
                energy_gains.append(energy_gain_per_swap)  # positive entries mean energy gain

            return energy_gains

        def expectation_value_of_dice(n):
            return (n + 1)/2

        energy_gains = compute_energy_gain_for_equal_env_swaps()
        features_as_index_lists = particle.getFeaturesAsIndexLists()
        n_distinct_environments = get_n_distinct_atomic_environments()

        expected_energy_gains = list()
        for symbol1_feature_index in range(n_distinct_environments):
            symbol2_feature_index = get_feature_index_of_other_element(symbol1_feature_index)
            max_swaps = min(len(features_as_index_lists[symbol1_feature_index]), len(features_as_index_lists[symbol2_feature_index]))
            expected_energy_gain = expectation_value_of_dice(max_swaps)*energy_gains[symbol1_feature_index]
            expected_energy_gains.append(expected_energy_gain)

        symbol1_feature_index = expected_energy_gains.index(max(expected_energy_gains))
        symbol2_feature_index = get_feature_index_of_other_element(symbol1_feature_index)
        max_swaps = min(len(features_as_index_lists[symbol1_feature_index]), len(features_as_index_lists[symbol2_feature_index]))
        n_swaps = np.random.randint(1, max_swaps + 1)  # randint upper limit is exclusive
        print(max_swaps)
        print(n_swaps)
        symbol1_indices = np.random.choice(features_as_index_lists[symbol1_feature_index], n_swaps, replace=False)
        symbol2_indices = np.random.choice(features_as_index_lists[symbol2_feature_index], n_swaps, replace=False)
        print(symbol1_indices)
        print(symbol2_indices)
        swaps = zip(symbol1_indices, symbol2_indices)
        particle.atoms.swapAtoms(swaps)

        return particle, swaps

    def mutation_crossover(self, parent1, parent2):  # this is for homotopes only!
        # find differences and equalities in the ordering
        parent_indices = parent1.atoms.getIndices()

        differences = list()
        equalitities = list()

        symbol1 = parent1.atoms.getSymbols()[0]
        symbol2 = parent1.atoms.getSymbols()[1]
        symbol1_occurences_equalities = 0
        for index in parent_indices:
            if parent1.atoms.getSymbol(index) == parent2.atoms.getSymbol(index):
                equalitities.append(index)
                if parent1.atoms.getSymbol(index) == symbol1:
                    symbol1_occurences_equalities += 1
            else:
                differences.append(index)

        if len(differences) == 0:
            print("Equal!")

        # distribute the atoms among the differences while keeping the stoichiometry fixed
        parent_stoichiometry = parent1.getStoichiometry()
        n_remaining_atoms_symbol1 = parent_stoichiometry[symbol1] - symbol1_occurences_equalities
        new_symbol1_indices = np.random.choice(differences, n_remaining_atoms_symbol1, replace=False)
        new_symbol2_indices = set(differences).difference(set(new_symbol1_indices))

        newParticle = copy.deepcopy(parent1)
        new_symbol1_atoms = zip(new_symbol1_indices, [symbol1]*len(new_symbol1_indices))
        new_symbol2_atoms = zip(new_symbol2_indices, [symbol2]*len(new_symbol2_indices))

        newParticle.atoms.transformAtoms(new_symbol1_atoms)
        newParticle.atoms.transformAtoms(new_symbol2_atoms)

        return newParticle
