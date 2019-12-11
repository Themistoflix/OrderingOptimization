import numpy as np


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
        symbol1_indices, symbol2_indices = zip(*swaps)
        particle.atoms.swapAtoms(zip(symbol2_indices, symbol1_indices))

        return particle
