import numpy as np
import copy

from ase import Atoms
from ase.optimize import BFGS
from asap3 import EMT

from Code.BoundingBox import BoundingBox
from Code.CuttingPlaneUtilities import CuttingPlane
from Code.IndexedAtoms import IndexedAtoms
from Code.NeighborList import NeighborList


class BaseNanoparticle:
    def __init__(self, lattice):
        self.lattice = lattice
        self.atoms = IndexedAtoms()
        self.neighborList = NeighborList(lattice)
        self.boundingBox = BoundingBox()

    def fromParticleData(self, atoms, neighborList=None):
        self.atoms = atoms
        if neighborList is None:
            self.constructNeighborList()
        else:
            self.neighborList = neighborList()

        self.constructBoundingBox()

    def addAtoms(self, atoms):
        self.atoms.addAtoms(atoms)
        indices, _ = zip(*atoms)
        self.neighborList.addAtoms(indices)

    def removeAtoms(self, latticeIndices):
        self.atoms.removeAtoms(latticeIndices)
        self.neighborList.removeAtoms(latticeIndices)

    def randomChemicalOrdering(self, symbols, atomsOfEachKind):
        self.atoms.randomChemicalOrdering(symbols, atomsOfEachKind)

    def rectangularPrism(self, w, l, h, symbol='X'):
        anchorPoint = self.lattice.getAnchorIndexOfCenteredBox(w, l, h)
        for x in range(w):
            for y in range(l):
                for z in range(h):
                    curPosition = anchorPoint + np.array([x, y, z])

                    if self.lattice.isValidLatticePosition(curPosition):
                        latticeIndex = self.lattice.getIndexFromLatticePosition(curPosition)
                        self.atoms.addAtoms([(latticeIndex, symbol)])
        self.constructNeighborList()

    def convexShape(self, numberOfAtomsOfEachKind, atomicSymbols, w, l, h, cuttingPlaneGenerator):
        self.rectangularPrism(w, l, h)
        self.constructBoundingBox()
        indicesOfCurrentAtoms = set(self.atoms.getIndices())

        finalNumberOfAtoms = sum(numberOfAtomsOfEachKind)
        MAX_CUTTING_ATTEMPTS = 50
        currentCuttingAttempt = 0
        cuttingPlaneGenerator.setCenter(self.boundingBox.get_center())

        while len(indicesOfCurrentAtoms) > finalNumberOfAtoms and currentCuttingAttempt < MAX_CUTTING_ATTEMPTS:
            # create cut plane
            cuttingPlane = cuttingPlaneGenerator.generateNewCuttingPlane()

            # count atoms to be removed, if new Count >= final Number remove
            atomsToBeRemoved, atomsToBeKept = cuttingPlane.splitAtomIndices(self.lattice, indicesOfCurrentAtoms)
            if len(atomsToBeRemoved) != 0.0 and len(indicesOfCurrentAtoms) - len(atomsToBeRemoved) >= finalNumberOfAtoms:
                indicesOfCurrentAtoms = indicesOfCurrentAtoms.difference(atomsToBeRemoved)
                currentCuttingAttempt = 0
            else:
                currentCuttingAttempt = currentCuttingAttempt + 1

        if currentCuttingAttempt == MAX_CUTTING_ATTEMPTS:
            # place cutting plane parallel to one of the axes and at the anchor point
            cuttingPlane = cuttingPlaneGenerator.createAxisParallelCuttingPlane(self.boundingBox.position)

            # shift till too many atoms would get removed
            numberOfAtomsYetToBeRemoved = len(indicesOfCurrentAtoms) - finalNumberOfAtoms
            atomsToBeRemoved = set()
            while len(atomsToBeRemoved) < numberOfAtomsYetToBeRemoved:
                cuttingPlane = CuttingPlane(cuttingPlane.anchor + cuttingPlane.normal * self.lattice.latticeConstant, cuttingPlane.normal)
                atomsToBeKept, atomsToBeRemoved = cuttingPlane.splitAtomIndices(self.lattice, indicesOfCurrentAtoms)

            # remove atoms till the final number is reached "from the ground up"

            # TODO implement sorting prioritzing the different directions in random order
            def sortByPosition(atom):
                return self.lattice.getLatticePositionFromIndex(atom)[0]

            atomsToBeRemoved = list(atomsToBeRemoved)
            atomsToBeRemoved.sort(key=sortByPosition)
            atomsToBeRemoved = atomsToBeRemoved[:numberOfAtomsYetToBeRemoved]

            atomsToBeRemoved = set(atomsToBeRemoved)
            indicesOfCurrentAtoms = indicesOfCurrentAtoms.difference(atomsToBeRemoved)

        # redistribute the different elements randomly
        self.atoms.clear()
        self.atoms.addAtoms(zip(indicesOfCurrentAtoms, ['X']*len(indicesOfCurrentAtoms)))
        self.atoms.randomChemicalOrdering(atomicSymbols, numberOfAtomsOfEachKind)

        self.constructNeighborList()

    def optimizeCoordinationNumbers(self, steps=15):
        for step in range(steps):
            outerAtoms = self.getAtomIndicesFromCoordinationNumbers(range(9))
            outerAtoms.sort(key=lambda x: self.getCoordinationNumber(x))

            startIndex = outerAtoms[0]
            symbol = self.atoms.getSymbol(startIndex)

            surfaceVacancies = list(self.getSurfaceVacancies())
            surfaceVacancies.sort(key=lambda x: self.getNumberOfAtomicNeighbors(x), reverse=True)

            endIndex = surfaceVacancies[0]

            self.removeAtoms([startIndex])
            self.addAtoms([(endIndex, symbol)])

    def constructNeighborList(self):
        self.neighborList.construct(self.atoms.getIndices())

    def constructBoundingBox(self):
        self.boundingBox.construct(self.lattice, self.atoms.getIndices())

    def getCornerAtomIndices(self, symbol=None):
        cornerCoordinationNumbers = [1, 2, 3, 4]
        return self.getAtomIndicesFromCoordinationNumbers(cornerCoordinationNumbers, symbol)

    def getEdgeIndices(self, symbol=None):
        edgeCoordinationNumbers = [5, 6, 7]
        return self.getAtomIndicesFromCoordinationNumbers(edgeCoordinationNumbers, symbol)

    def getSurfaceAtomIndices(self, symbol=None):
        surfaceCoordinationNumbers = [8, 9]
        return self.getAtomIndicesFromCoordinationNumbers(surfaceCoordinationNumbers, symbol)

    def getTerraceAtomIndices(self, symbol=None):
        terraceCoordinationNumbers = [10, 11]
        return self.getAtomIndicesFromCoordinationNumbers(terraceCoordinationNumbers, symbol)

    def getInnerAtomIndices(self, symbol=None):
        innerCoordinationNumbers = [12]
        return self.getAtomIndicesFromCoordinationNumbers(innerCoordinationNumbers, symbol)

    def getNumberOfHeteroatomicBonds(self):
        numberOfHeteroatomicBonds = 0

        if len(self.atoms.getSymbols()) == 2:
            symbol = self.atoms.getSymbols()[0]
        else:
            return 0

        for latticeIndexWithSymbol in self.atoms.getIndicesBySymbol(symbol):
            neighborList = self.neighborList[latticeIndexWithSymbol]
            for neighbor in neighborList:
                symbolOfNeighbor = self.atoms.getSymbol(neighbor)

                if symbol != symbolOfNeighbor:
                    numberOfHeteroatomicBonds = numberOfHeteroatomicBonds + 1

        return numberOfHeteroatomicBonds

    def getAtomIndicesFromCoordinationNumbers(self, coordinationNumbers, symbol=None):
        if symbol is None:
            return list(filter(lambda x: self.getCoordinationNumber(x) in coordinationNumbers, self.atoms.getIndices()))
        else:
            return list(filter(lambda x: self.getCoordinationNumber(x) in coordinationNumbers and self.atoms.getSymbol(x) == symbol, self.atoms.getIndices()))

    def getCoordinationNumber(self, latticeIndex):
        return self.neighborList.getCoordinationNumber(latticeIndex)

    def getAtoms(self, atomIndices=None):
        return copy.deepcopy(self.atoms.getAtoms(atomIndices))

    def getNeighborList(self):
        return self.neighborList

    def getASEAtoms(self, centered=True):
        atomPositions = list()
        atomicSymbols = list()
        for latticeIndex in self.atoms.getIndices():
            atomPositions.append(self.lattice.getCartesianPositionFromIndex(latticeIndex))
            atomicSymbols.append(self.atoms.getSymbol(latticeIndex))

        atoms = Atoms(positions=atomPositions, symbols=atomicSymbols)
        if centered:
            COM = atoms.get_center_of_mass()
            return Atoms(positions=[position - COM for position in atomPositions], symbols=atomicSymbols)
        else:
            return Atoms(positions=atomPositions, symbols=atomicSymbols)

    def getPotentialEnergy(self, steps=100):
        cellWidth = self.lattice.width*self.lattice.latticeConstant
        cellLength = self.lattice.length*self.lattice.latticeConstant
        cellHeight = self.lattice.height*self.lattice.latticeConstant

        atoms = self.getASEAtoms()
        atoms.set_cell(np.array([[cellWidth, 0, 0], [0, cellLength, 0], [0, 0, cellHeight]]))
        atoms.set_calculator(EMT())
        dyn = BFGS(atoms)
        dyn.run(fmax=0.01, steps=steps)

        return atoms.get_potential_energy()

    def getStoichiometry(self):
        return self.atoms.getStoichiometry()

    def getSurfaceVacancies(self):
        notFullyCoordinatedAtoms = self.getAtomIndicesFromCoordinationNumbers(range(self.lattice.MAX_NEIGHBORS))
        surfaceVacancies = set()

        for atom in notFullyCoordinatedAtoms:
            neighborVacancies = self.lattice.getNearestNeighbors(atom).difference(self.neighborList[atom])
            surfaceVacancies = surfaceVacancies.union(neighborVacancies)
        return surfaceVacancies

    def getAtomicNeighbors(self, index):
        neighbors = list()
        nearestNeighbors = self.lattice.getNearestNeighbors(index)
        for latticeIndex in nearestNeighbors:
            if latticeIndex in self.atoms.getIndices():
                neighbors.append(latticeIndex)

        return neighbors

    def getNumberOfAtomicNeighbors(self, index):
        return len(self.getAtomicNeighbors(index))