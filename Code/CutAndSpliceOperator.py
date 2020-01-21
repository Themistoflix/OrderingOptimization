from Code import Nanoparticle as NP
from Code.IndexedAtoms import IndexedAtoms


class CutAndSpliceOperator:
    def __init__(self, cutting_plane_generator):
        self.cutting_plane_generator = cutting_plane_generator

    def cut_and_splice(self, particle1, particle2):
        self.cutting_plane_generator.setCenter(particle1.boundingBox.get_center())
        common_lattice = particle1.lattice

        # make sure that we actually cut
        while True:
            cutting_plane = self.cutting_plane_generator.generateNewCuttingPlane()
            atom_indices_in_positive_subspace, _ = cutting_plane.splitAtomIndices(common_lattice, particle1.atoms.get_indices())
            _, atom_indices_in_negative_subspace = cutting_plane.splitAtomIndices(common_lattice, particle2.atoms.get_indices())

            if len(atom_indices_in_negative_subspace) > 0 and len(atom_indices_in_positive_subspace) > 0:
                break

        new_atom_data = IndexedAtoms()
        new_atom_data.add_atoms(particle1.get_atoms(atom_indices_in_positive_subspace))
        new_atom_data.add_atoms(particle2.get_atoms(atom_indices_in_negative_subspace))

        new_particle = NP.Nanoparticle(common_lattice, particle1.getL_max())
        new_particle.from_particle_data(new_atom_data)

        # old_stoichiometry = particle1.getStoichiometry()
        # new_particle.enforceStoichiometry(old_stoichiometry)
        return new_particle


