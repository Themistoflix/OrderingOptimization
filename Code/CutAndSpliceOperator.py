from Code import Nanoparticle as NP
from Code.IndexedAtoms import IndexedAtoms


class CutAndSpliceOperator:
    def __init__(self, cuttingPlaneGenerator):
        self.cuttingPlaneGenerator = cuttingPlaneGenerator

    def cut_and_splice(self, particle1, particle2):
        self.cuttingPlaneGenerator.setCenter(particle1.boundingBox.get_center())
        cutting_plane = self.cuttingPlaneGenerator.generateNewCuttingPlane()
        common_lattice = particle1.lattice

        atom_indices_in_positive_subspace, _ = cutting_plane.splitAtomIndices(common_lattice, particle1.atoms.getIndices())
        _, atom_indices_in_negative_subspace = cutting_plane.splitAtomIndices(common_lattice, particle2.atoms.getIndices())

        new_atom_data = IndexedAtoms()
        new_atom_data.addAtoms(particle1.getAtoms(atom_indices_in_positive_subspace))
        new_atom_data.addAtoms(particle2.getAtoms(atom_indices_in_negative_subspace))

        new_particle = NP.Nanoparticle(common_lattice, particle1.getL_max())
        new_particle.fromParticleData(new_atom_data)

        old_stoichiometry = particle1.getStoichiometry()
        print(new_particle.getStoichiometry())
        new_particle.enforceStoichiometry(old_stoichiometry)
        return new_particle


