import numpy as np
import copy


class MutationOperator:
    def __init__(self, p):
        self.p = p

    def random_mutation(self, particle):
        new_particle = copy.deepcopy(particle)

        symbols = np.random.choice(new_particle.atoms.get_symbols(), 2, replace=False)
        symbol_from = symbols[0]
        symbol_to = symbols[1]

        print("Symbol_ from:{0}".format(symbol_from))
        print("Symbol_ to:{0}".format(symbol_to))

        n_mutations = min(self.draw_from_geometric_distribution(), len(new_particle.atoms.get_indices_by_symbol(symbol_from)) - 1)

        atom_indices_to_be_transformed = np.random.choice(new_particle.atoms.get_indices_by_symbol(symbol_from), n_mutations, replace=False)

        print("atom indices: {0}".format(atom_indices_to_be_transformed))
        new_particle.atoms.transform_atoms(zip(atom_indices_to_be_transformed, [symbol_to] * n_mutations))

        return new_particle

    def draw_from_geometric_distribution(self):
        return np.random.geometric(p=self.p, size=1)[0]
