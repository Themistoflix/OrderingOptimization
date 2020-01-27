import random
import numpy as np


class CuttingPlane:
    def __init__(self, anchor, normal):
        self.anchor = anchor
        self.normal = normal

    def splitAtomIndices(self, lattice, atomIndices):
        atomsInPositiveSubspace = set()
        atomsInNegativeSubspace = set()

        for latticeIndex in atomIndices:
            position = lattice.get_cartesian_position_from_index(latticeIndex)
            if np.dot((position - self.anchor), self.normal) >= 0.0:
                atomsInPositiveSubspace.add(latticeIndex)
            else:
                atomsInNegativeSubspace.add(latticeIndex)
        return atomsInPositiveSubspace, atomsInNegativeSubspace


class CuttingPlaneGenerator:
    def __init__(self, center):
        self.center = center

    def generate_new_cutting_plane(self):
        raise NotImplementedError()

    def set_center(self, center):
        self.center = center

    def create_axis_parallel_cutting_plane(self, position):
        anchor = position
        normal = np.array([0, 0, 0])
        normal[random.randrange(2)] = 1.0

        return CuttingPlane(anchor, normal)


class SphericalCuttingPlaneGenerator(CuttingPlaneGenerator):
    def __init__(self, minRadius, maxRadius, center=0.0):
        super().__init__(center)
        self.minRadius = minRadius
        self.maxRadius = maxRadius

    def generate_new_cutting_plane(self):
        normal = np.array([random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1])
        normal = normal / np.linalg.norm(normal)
        anchor = normal * (self.minRadius + random.random() * (self.maxRadius - self.minRadius))
        anchor = anchor + self.center

        return CuttingPlane(anchor, normal)


