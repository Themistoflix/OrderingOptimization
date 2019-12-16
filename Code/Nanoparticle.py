from scipy.special import sph_harm
import numpy as np
import math

from Code.BaseNanoparticle import BaseNanoparticle


class Nanoparticle(BaseNanoparticle):
    def __init__(self, lattice, l_max):
        BaseNanoparticle.__init__(self, lattice)

        self.EMTEnergy = math.inf
        self.RREnergy = math.inf

        self.bondParameters = dict()
        self.featuresAsIndexLists = list()
        self.featureVector = np.array([])
        self.l_max = l_max

    def getL_max(self):
        return self.l_max

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

            neighborIndices = self.getAtomicNeighbors(latticeIndex)
            atomicSymbols = np.array([self.atoms.getSymbol(index) for index in neighborIndices])
            cartesianCoords = [self.lattice.getCartesianPositionFromIndex(index) for index in neighborIndices]

            centerAtomPosition = self.lattice.getCartesianPositionFromIndex(latticeIndex)
            cartesianCoords = list(map(lambda x: x - centerAtomPosition, cartesianCoords))

            angularCoordinates = mapOntoUnitSphere(cartesianCoords)

            # The density of each species is expanded separately
            expansionCoefficients = []
            numberOfNeighbors = len(atomicSymbols)
            for element in sorted(self.atoms.getSymbols()):
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
        for elementIndex, element in enumerate(sorted(self.atoms.getSymbols())):
            N_elementAtoms = len(list(filter(lambda x: self.atoms.getSymbol(x) == element, self.getAtomicNeighbors(latticeIndex))))
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
        for latticeIndex in self.atoms.getIndices():
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
        self.EMTEnergy = self.getPotentialEnergy(steps)
        return self.EMTEnergy

    def getEMTEnergy(self):
        return self.EMTEnergy

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
        atomNumberDifference = self.atoms.getCount() - sum(stoichiometry.values())

        if atomNumberDifference > 0:
            self.removeUndercoordinatedAtoms(atomNumberDifference)

        elif atomNumberDifference < 0:
            self.fillHighCoordinatedSurfaceVacancies(-atomNumberDifference)
            
        if self.getStoichiometry() != stoichiometry:
            self.adjustAtomicRatios(stoichiometry)

    def fillHighCoordinatedSurfaceVacancies(self, count):
        for atom in range(count):
            surfaceVacancies = list(self.getSurfaceVacancies())
            surfaceVacancies.sort(key=self.getNumberOfAtomicNeighbors, reverse=True)

            index = surfaceVacancies[0]
            symbol = np.random.choice(self.atoms.getSymbols(), 1)[0]

            self.addAtoms([(index, symbol)])

    def removeUndercoordinatedAtoms(self, count):
        for atom in range(count):
            mostUndercoordinatedAtoms = self.getAtomIndicesFromCoordinationNumbers(range(9))
            mostUndercoordinatedAtoms.sort(key=self.getCoordinationNumber)

            index = mostUndercoordinatedAtoms[0]
            self.removeAtoms([index])

    def adjustAtomicRatios(self, stoichiometry):
        element1 = self.atoms.getSymbols()[0]
        element2 = self.atoms.getSymbols()[1]
        n_differences_element1 = self.getStoichiometry()[element1] - stoichiometry[element1]

        if n_differences_element1 < 0:
            indices_to_transform_to_element1 = np.random.choice(self.atoms.getIndicesBySymbol(element2), -n_differences_element1, replace=False)
            self.atoms.transformAtoms(zip(indices_to_transform_to_element1, [element1]*(-n_differences_element1)))
        elif n_differences_element1 > 0:
            indices_to_transform_to_element2 = np.random.choice(self.atoms.getIndicesBySymbol(element1), n_differences_element1, replace=False)
            self.atoms.transformAtoms(zip(indices_to_transform_to_element2, [element2]*n_differences_element1))


