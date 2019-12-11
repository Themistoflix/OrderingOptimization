class NeighborList:
    def __init__(self, lattice):
        self.list = dict()
        self.lattice = lattice

    def __getitem__(self, item):
        return self.list[item]

    def __setitem__(self, key, value):
        self.list[key] = value

    def construct(self, latticeIndices):
        for latticeIndex in latticeIndices:
            nearestLatticeNeighbors = self.lattice.getNearestNeighbors(latticeIndex)
            nearestNeighbors = set()
            for neighbor in nearestLatticeNeighbors:
                if neighbor in latticeIndices:
                    nearestNeighbors.add(neighbor)

            self.list[latticeIndex] = nearestNeighbors

    def addAtoms(self, latticeIndices):
        allAtoms = list(self.list.keys())
        for latticeIndex in latticeIndices:
            nearestNeighbors = set()
            nearestLatticeNeighbors = self.lattice.getNearestNeighbors(latticeIndex)
            for neighbor in nearestLatticeNeighbors:
                if neighbor in allAtoms:
                    nearestNeighbors.add(neighbor)
                    self.list[neighbor].add(latticeIndex)

            self.list[latticeIndex] = nearestNeighbors

    def removeAtoms(self, latticeIndices):
        for latticeIndex in latticeIndices:
            neighbors = self.list[latticeIndex]
            for neighbor in neighbors:
                self.list[neighbor].remove(latticeIndex)

            del self.list[latticeIndex]

    def getCoordinationNumber(self, latticeIndex):
        return len(self.list[latticeIndex])
