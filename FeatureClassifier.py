import numpy as np
from sklearn.cluster import KMeans


class FeatureClassifier:
    def __init__(self):
        pass

    def compute_features_as_index_list(self, particle, recompute_bond_parameters=False):
        if recompute_bond_parameters:
            particle.computeBondParameters()

        n_features = self.compute_n_features(particle)

        features_as_index_lists = list()  # need empty list in case of recalculation
        for i in range(n_features):
            l = list()
            features_as_index_lists.append(l)

        for atomIndex in particle.atoms.getIndices():
            feature = self.predict_atom_feature(particle, atomIndex)
            features_as_index_lists[feature].append(atomIndex)

        particle.setFeaturesAsIndexLists(features_as_index_lists)

    def compute_feature_vector(self, particle, recompute_bond_parameters=False):
        self.compute_features_as_index_list(particle, recompute_bond_parameters)

        n_features = self.compute_n_features(particle)
        feature_vector = np.array([len(particle.getFeaturesAsIndexLists()[feature]) for feature in range(n_features)])

        particle.setFeatureVector(feature_vector)

    def compute_n_features(self, particle):
        raise NotImplementedError

    def predict_atom_feature(self, particle, latticeIndex, recomputeBondParameter=False):
        raise NotImplementedError

    def train(self, training_set):
        raise NotImplementedError


class KMeansClassifier(FeatureClassifier):
    def __init__(self, n_cluster):
        FeatureClassifier.__init__(self)
        self.kMeans = None
        self.n_cluster = n_cluster

    def compute_n_features(self, particle):
        n_elements = len(particle.atoms.getSymbols())
        n_features = self.n_cluster * n_elements
        return n_features

    def predict_atom_feature(self, particle, latticeIndex, recomputeBondParameter=False):
        symbol = particle.atoms.getSymbol(latticeIndex)
        symbols = sorted(particle.atoms.getSymbols())
        symbol_index = symbols.index(symbol)

        offset = symbol_index*self.n_cluster
        if recomputeBondParameter:
            environment = self.kMeans.predict([particle.computeBondParameter(latticeIndex)])[0]
        else:
            environment = self.kMeans.predict([particle.getBondParameter(latticeIndex)])[0]
        return offset + environment

    def train(self, training_set):
        bond_parameters = list()
        for particle in training_set:
            bond_parameters = bond_parameters + particle.getBondParameters()

        self.kMeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(bond_parameters)


class PreClusteringClassifier(FeatureClassifier):
    def __init__(self, cluster_sizes):
        FeatureClassifier.__init__(self)
        self.cluster_sizes = cluster_sizes
        self.kMeans_list = [KMeans(n_clusters=n_cluster, random_state=0) for n_cluster in self.cluster_sizes]
        self.coordination_number_offset = np.cumsum(np.array([0] + self.cluster_sizes[:-1]))
        self.distinct_environments = sum(self.cluster_sizes)

    def compute_n_features(self, particle):
        n_elements = len(particle.atoms.getSymbols())
        n_features = n_elements * self.distinct_environments
        return n_features

    def get_element_offset_multiplier(self, all_symbols, symbol):
        all_symbols = sorted(all_symbols)
        return all_symbols.index(symbol)

    def predict_atom_feature(self, particle, atomIndex, recomputeBondParameter=False):
        symbol = particle.atoms.getSymbol(atomIndex)
        coordination_number = particle.getCoordinationNumber(atomIndex)
        if recomputeBondParameter:
            label = self.kMeans_list[coordination_number - 1].predict([particle.computeBondParameter(atomIndex)])[0]
        else:
            label = self.kMeans_list[coordination_number - 1].predict([particle.getBondParameter(atomIndex)])[0]
        element_offset_multiplier = self.get_element_offset_multiplier(particle.atoms.getSymbols(), symbol)

        offset = element_offset_multiplier * self.distinct_environments + self.coordination_number_offset[coordination_number - 1]

        return offset + label

    def train(self, training_set):
        for coordination_number in range(1, 13):
            bond_parameters = list()
            for particle in training_set:
                indices = particle.getAtomIndicesFromCoordinationNumbers([coordination_number])
                for index in indices:
                    bond_parameters = bond_parameters + [particle.getBondParameter(index)]
            print("Coordination number: {0}".format(coordination_number))
            print("Environment count: {0}".format(len(bond_parameters)))

            self.kMeans_list[coordination_number - 1].fit(bond_parameters)
