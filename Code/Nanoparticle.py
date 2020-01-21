from scipy.special import sph_harm
import numpy as np
import math

from Code.BaseNanoparticle import BaseNanoparticle


class Nanoparticle(BaseNanoparticle):
    def __init__(self, lattice, l_max):
        BaseNanoparticle.__init__(self, lattice)

        self.energies = dict()
        self.EMTEnergy = math.inf
        self.RREnergy = math.inf

        self.bondParameters = dict()
        self.featuresAsIndexLists = list()
        self.featureVector = np.array([])
        self.l_max = l_max

        self.mixingParameters = dict()

    def getL_max(self):
        return self.l_max

    def set_energy(self, key, energy):
        self.energies[key] = energy

    def get_energy(self, key):
        return self.energies[key]

    def computeBondParameter(self, latticeIndex):
        def mapOntoUnitSphere(cartesianCoords):
            # note the use of the scipy.special.sph_harm notation for phi and theta (which is the opposite of wikipedias)
            def angularFromCartesianCoords(cartesianCoords):
                x = cartesianCoords[0]
                y = cartesianCoords[1]
                z = cartesianCoords[2]

                hxy = np.hypot(x, y)
                r = np.hypot(hxy, z)
                el = np.arctan2(z, hxy)
                az = np.arctan2(y, x)
                return np.abs(az + np.pi), np.abs(el + np.pi / 2.0)

            return list(map(lambda x: angularFromCartesianCoords(x), cartesianCoords))

        def sphericalHarmonicsExpansion():
            """
            This functions takes the environment atoms surrounding a reference atom
            in an fcc lattice and returns the spherical harmonic coefficients of the expansion
            """

            neighborIndices = self.get_atomic_neighbors(latticeIndex)
            atomicSymbols = np.array([self.atoms.get_symbol(index) for index in neighborIndices])
            cartesianCoords = [self.lattice.getCartesianPositionFromIndex(index) for index in neighborIndices]

            centerAtomPosition = self.lattice.getCartesianPositionFromIndex(latticeIndex)
            cartesianCoords = list(map(lambda x: x - centerAtomPosition, cartesianCoords))

            angularCoordinates = mapOntoUnitSphere(cartesianCoords)

            # The density of each species is expanded separately
            expansionCoefficients = []
            numberOfNeighbors = len(atomicSymbols)
            for element in sorted(self.atoms.get_symbols()):
                elementDensity = np.zeros(numberOfNeighbors)
                elementDensity[np.where(atomicSymbols == element)] += 1
                Clms_element = []
                for l in range(self.l_max + 1):
                    for m in range(-l, l + 1):
                        Clm = 0.0
                        for i, point in enumerate(angularCoordinates):
                            if elementDensity[i] != 0:
                                Clm += elementDensity[i] * np.conj(sph_harm(m, l, point[0], point[1]))
                        Clms_element.append(Clm)
                expansionCoefficients.append(Clms_element)
            return expansionCoefficients

        SHExpansionCoefficients = sphericalHarmonicsExpansion()
        bondParameters = []
        for elementIndex, element in enumerate(sorted(self.atoms.get_symbols())):
            N_elementAtoms = len(list(filter(lambda x: self.atoms.get_symbol(x) == element, self.get_atomic_neighbors(latticeIndex))))
            Qls_element = []
            i = 0
            for l in range(self.l_max + 1):
                Ql = 0
                if N_elementAtoms != 0:
                    for m in range(-l, l + 1):
                        Ql += 1.0 / (N_elementAtoms ** 2) * np.conj(SHExpansionCoefficients[elementIndex][i]) * SHExpansionCoefficients[elementIndex][i]
                        i += 1
                Qls_element.append(np.sqrt((np.sqrt(4.0*np.pi)/(2.*l+1.))*Ql))
            bondParameters.append(Qls_element)

        bondParameters = np.array(bondParameters)
        bondParameters = bondParameters.real
        bondParameters = np.reshape(bondParameters, bondParameters.size) # return the bond parameters as one big feature vector
        return bondParameters

    def getBondParameter(self, latticeIndex):
        return self.bondParameters[latticeIndex]

    def computeBondParameters(self):
        for latticeIndex in self.atoms.get_indices():
            self.bondParameters[latticeIndex] = self.computeBondParameter(latticeIndex)

    def getBondParameters(self):
        return list(self.bondParameters.values())

    def getNumberOfFeatures(self):
        return len(self.featuresAsIndexLists)

    def setFeaturesAsIndexLists(self, featureAsIndexLists):
        self.featuresAsIndexLists = featureAsIndexLists

    def getFeaturesAsIndexLists(self):
        return self.featuresAsIndexLists

    def computeEMTEnergy(self, steps):
        self.EMTEnergy = self.get_potential_energy(steps)
        return self.EMTEnergy

    def getEMTEnergy(self):
        return self.EMTEnergy

    def computeMixingParameters(self):
        for symbol in self.get_stoichiometry():
            pureParticle = Nanoparticle(self.lattice, self.l_max)
            pureParticle.add_atoms(list(zip(self.atoms.get_indices(), [symbol] * self.atoms.get_n_atoms())))
            pureParticle.computeEMTEnergy(20)
            self.mixingParameters[symbol] = pureParticle.getEMTEnergy()

        return self.mixingParameters

    def setMixingParameters(self, mixingParameters):
        self.mixingParameters = mixingParameters

    def getMixingParameters(self):
        return self.mixingParameters

    def getMixingEnergy(self):
        E_mixing = self.EMTEnergy
        numberOfAtoms = self.atoms.get_n_atoms()

        for symbol in self.get_stoichiometry():
            E_mixing -= self.mixingParameters[symbol] * self.get_stoichiometry()[symbol] / numberOfAtoms

        return E_mixing

    def computeRREnergy(self, ridge):
        self.RREnergy = np.dot(np.transpose(ridge.coef_), self.getFeatureVector())

    def getRREnergy(self):
        return self.RREnergy

    def setFeatureVector(self, featureVector):
        self.featureVector = featureVector

    def getFeatureVector(self):
        return self.featureVector

    def getFeatureOfAtom(self, latticeIndex):
        for feature, latticeIndices in enumerate(self.featuresAsIndexLists):
            if latticeIndex in latticeIndices:
                return feature

    def enforceStoichiometry(self, stoichiometry):
        atomNumberDifference = self.atoms.get_n_atoms() - sum(stoichiometry.values())

        if atomNumberDifference > 0:
            self.removeUndercoordinatedAtoms(atomNumberDifference)

        elif atomNumberDifference < 0:
            self.fillHighCoordinatedSurfaceVacancies(-atomNumberDifference)
            
        if self.get_stoichiometry() != stoichiometry:
            self.adjustAtomicRatios(stoichiometry)

    def fillHighCoordinatedSurfaceVacancies(self, count):
        for atom in range(count):
            surfaceVacancies = list(self.get_surface_vacancies())
            surfaceVacancies.sort(key=self.get_n_atomic_neighbors, reverse=True)

            index = surfaceVacancies[0]
            symbol = np.random.choice(self.atoms.get_symbols(), 1)[0]

            self.add_atoms([(index, symbol)])

    def removeUndercoordinatedAtoms(self, count):
        for atom in range(count):
            mostUndercoordinatedAtoms = self.get_atom_indices_from_coordination_number(range(9))
            mostUndercoordinatedAtoms.sort(key=self.get_coordination_number)

            index = mostUndercoordinatedAtoms[0]
            self.remove_atoms([index])

    def adjustAtomicRatios(self, stoichiometry):
        element1 = self.atoms.get_symbols()[0]
        element2 = self.atoms.get_symbols()[1]
        n_differences_element1 = self.get_stoichiometry()[element1] - stoichiometry[element1]

        if n_differences_element1 < 0:
            indices_to_transform_to_element1 = np.random.choice(self.atoms.get_indices_by_symbol(element2), -n_differences_element1, replace=False)
            self.atoms.transform_atoms(zip(indices_to_transform_to_element1, [element1] * (-n_differences_element1)))
        elif n_differences_element1 > 0:
            indices_to_transform_to_element2 = np.random.choice(self.atoms.get_indices_by_symbol(element1), n_differences_element1, replace=False)
            self.atoms.transform_atoms(zip(indices_to_transform_to_element2, [element2] * n_differences_element1))

    def octahedron(self, height, symbol):
        boundingBoxAnchor = self.lattice.getAnchorIndexOfCenteredBox(2 * height, 2 * height, 2 * height)
        lowerTipPosition = boundingBoxAnchor + np.array([height, height, 0])

        if not self.lattice.isValidLatticePosition(lowerTipPosition):
            lowerTipPosition[2] = lowerTipPosition[2] + 1

        layerBasisVector1 = np.array([1, 1, 0])
        layerBasisVector2 = np.array([-1, 1, 0])
        for zPosition in range(height):
            layerWidth = zPosition + 1
            lowerLayerOffset = np.array([0, -zPosition, zPosition])
            upperLayerOffset = np.array([0, -zPosition, 2 * height - 2 - zPosition])

            lowerLayerStartPosition = lowerTipPosition + lowerLayerOffset
            upperLayerStartPosition = lowerTipPosition + upperLayerOffset
            for width in range(layerWidth):
                for length in range(layerWidth):
                    currentPositionLowerLayer = lowerLayerStartPosition + width * layerBasisVector1 + length * layerBasisVector2
                    currentPositionUpperLayer = upperLayerStartPosition + width * layerBasisVector1 + length * layerBasisVector2

                    lowerLayerIndex = self.lattice.getIndexFromLatticePosition(currentPositionLowerLayer)
                    upperLayerIndex = self.lattice.getIndexFromLatticePosition(currentPositionUpperLayer)

                    self.atoms.add_atoms([(lowerLayerIndex, symbol), (upperLayerIndex, symbol)])

        self.construct_neighbor_list()

        self.construct_bounding_box()

    def kozlovSphere(self, height, symbols, numberOfAtomsEachKind):
        boundingBoxAnchor = self.lattice.getAnchorIndexOfCenteredBox(2 * height, 2 * height, 2 * height)
        lowerTipPosition = boundingBoxAnchor + np.array([height, height, 0])

        if not self.lattice.isValidLatticePosition(lowerTipPosition):
            lowerTipPosition[2] = lowerTipPosition[2] + 1

        layerBasisVector1 = np.array([1, 1, 0])
        layerBasisVector2 = np.array([-1, 1, 0])
        for zPosition in range(height):
            layerWidth = zPosition + 1
            lowerLayerOffset = np.array([0, -zPosition, zPosition])
            upperLayerOffset = np.array([0, -zPosition, 2 * height - 2 - zPosition])

            lowerLayerStartPosition = lowerTipPosition + lowerLayerOffset
            upperLayerStartPosition = lowerTipPosition + upperLayerOffset
            for width in range(layerWidth):
                for length in range(layerWidth):
                    currentPositionLowerLayer = lowerLayerStartPosition + width * layerBasisVector1 + length * layerBasisVector2
                    currentPositionUpperLayer = upperLayerStartPosition + width * layerBasisVector1 + length * layerBasisVector2

                    lowerLayerIndex = self.lattice.getIndexFromLatticePosition(currentPositionLowerLayer)
                    upperLayerIndex = self.lattice.getIndexFromLatticePosition(currentPositionUpperLayer)

                    self.atoms.add_atoms([(lowerLayerIndex, 'X'), (upperLayerIndex, 'X')])

        self.construct_neighbor_list()
        corners = self.get_atom_indices_from_coordination_number([4])

        self.remove_atoms(corners)
        self.random_ordering(symbols, numberOfAtomsEachKind)

