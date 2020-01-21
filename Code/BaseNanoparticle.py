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
        self.neighbor_list = NeighborList(lattice)
        self.bounding_box = BoundingBox()

    def from_particle_data(self, atoms, neighbor_list=None):
        self.atoms = atoms
        if neighbor_list is None:
            self.construct_neighbor_list()
        else:
            self.neighbor_list = neighbor_list()

        self.construct_bounding_box()

    def add_atoms(self, atoms):
        self.atoms.add_atoms(atoms)
        indices, _ = zip(*atoms)
        self.neighbor_list.addAtoms(indices)

    def remove_atoms(self, latticeIndices):
        self.atoms.remove_atoms(latticeIndices)
        self.neighbor_list.removeAtoms(latticeIndices)

    def random_ordering(self, symbols, n_atoms_same_symbol):
        self.atoms.random_ordering(symbols, n_atoms_same_symbol)

    def rectangular_prism(self, w, l, h, symbol='X'):
        anchor_point = self.lattice.getAnchorIndexOfCenteredBox(w, l, h)
        for x in range(w):
            for y in range(l):
                for z in range(h):
                    cur_position = anchor_point + np.array([x, y, z])

                    if self.lattice.isValidLatticePosition(cur_position):
                        lattice_index = self.lattice.getIndexFromLatticePosition(cur_position)
                        self.atoms.add_atoms([(lattice_index, symbol)])
        self.construct_neighbor_list()

    def convex_shape(self, symbols, n_atoms_same_symbol, w, l, h, cutting_plane_generator):
        self.rectangular_prism(w, l, h)
        self.construct_bounding_box()
        indices_current_atoms = set(self.atoms.get_indices())

        final_n_atoms = sum(n_atoms_same_symbol)
        MAX_CUTTING_ATTEMPTS = 50
        cur_cutting_attempt = 0
        cutting_plane_generator.setCenter(self.bounding_box.get_center())

        while len(indices_current_atoms) > final_n_atoms and cur_cutting_attempt < MAX_CUTTING_ATTEMPTS:
            # create cut plane
            cutting_plane = cutting_plane_generator.generateNewCuttingPlane()

            # count atoms to be removed, if new Count >= final Number remove
            atoms_to_be_removed, atoms_to_be_kept = cutting_plane.splitAtomIndices(self.lattice, indices_current_atoms)
            if len(atoms_to_be_removed) != 0.0 and len(indices_current_atoms) - len(atoms_to_be_removed) >= final_n_atoms:
                indices_current_atoms = indices_current_atoms.difference(atoms_to_be_removed)
                cur_cutting_attempt = 0
            else:
                cur_cutting_attempt = cur_cutting_attempt + 1

        if cur_cutting_attempt == MAX_CUTTING_ATTEMPTS:
            # place cutting plane parallel to one of the axes and at the anchor point
            cutting_plane = cutting_plane_generator.createAxisParallelCuttingPlane(self.bounding_box.position)

            # shift till too many atoms would get removed
            n_atoms_yet_to_be_removed = len(indices_current_atoms) - final_n_atoms
            atoms_to_be_removed = set()
            while len(atoms_to_be_removed) < n_atoms_yet_to_be_removed:
                cutting_plane = CuttingPlane(cutting_plane.anchor + cutting_plane.normal * self.lattice.latticeConstant, cutting_plane.normal)
                atoms_to_be_kept, atoms_to_be_removed = cutting_plane.splitAtomIndices(self.lattice, indices_current_atoms)

            # remove atoms till the final number is reached "from the ground up"

            # TODO implement sorting prioritzing the different directions in random order
            def sort_by_position(atom):
                return self.lattice.getLatticePositionFromIndex(atom)[0]

            atoms_to_be_removed = list(atoms_to_be_removed)
            atoms_to_be_removed.sort(key=sort_by_position)
            atoms_to_be_removed = atoms_to_be_removed[:n_atoms_yet_to_be_removed]

            atoms_to_be_removed = set(atoms_to_be_removed)
            indices_current_atoms = indices_current_atoms.difference(atoms_to_be_removed)

        # redistribute the different elements randomly
        self.atoms.clear()
        self.atoms.add_atoms(zip(indices_current_atoms, ['X'] * len(indices_current_atoms)))
        self.atoms.random_ordering(symbols, n_atoms_same_symbol)

        self.construct_neighbor_list()

    def optimize_coordination_numbers(self, steps=15):
        for step in range(steps):
            outer_atoms = self.get_atom_indices_from_coordination_number(range(9))
            outer_atoms.sort(key=lambda x: self.get_coordination_number(x))

            start_index = outer_atoms[0]
            symbol = self.atoms.get_symbol(start_index)

            surface_vacancies = list(self.get_surface_vacancies())
            surface_vacancies.sort(key=lambda x: self.get_n_atomic_neighbors(x), reverse=True)

            end_index = surface_vacancies[0]
            if self.get_coordination_number(start_index) < self.get_n_atomic_neighbors(end_index):
                self.remove_atoms([start_index])
                self.add_atoms([(end_index, symbol)])
            else:
                break

        self.construct_neighbor_list()

    def construct_neighbor_list(self):
        self.neighbor_list.construct(self.atoms.get_indices())

    def construct_bounding_box(self):
        self.bounding_box.construct(self.lattice, self.atoms.get_indices())

    def get_corner_atom_indices(self, symbol=None):
        corner_coordination_numbers = [1, 2, 3, 4]
        return self.get_atom_indices_from_coordination_number(corner_coordination_numbers, symbol)

    def get_edge_indices(self, symbol=None):
        edge_coordination_numbers = [5, 6, 7]
        return self.get_atom_indices_from_coordination_number(edge_coordination_numbers, symbol)

    def get_surface_atom_indices(self, symbol=None):
        surface_coordination_numbers = [8, 9]
        return self.get_atom_indices_from_coordination_number(surface_coordination_numbers, symbol)

    def get_terrace_atom_indices(self, symbol=None):
        terrace_coordination_numbers = [10, 11]
        return self.get_atom_indices_from_coordination_number(terrace_coordination_numbers, symbol)

    def get_inner_atom_indices(self, symbol=None):
        innerCoordinationNumbers = [12]
        return self.get_atom_indices_from_coordination_number(innerCoordinationNumbers, symbol)

    def get_n_homo_and_heterogenous_bonds(self):
        n_heteroatomic_bonds = 0
        n_homogenous_bonds = 0

        symbol = self.atoms.get_symbols()[0]

        for lattice_index_with_symbol in self.atoms.get_indices_by_symbol(symbol):
            neighbor_list = self.neighbor_list[lattice_index_with_symbol]
            for neighbor in neighbor_list:
                symbol_of_neighbor = self.atoms.get_symbol(neighbor)

                if symbol != symbol_of_neighbor:
                    n_heteroatomic_bonds = n_heteroatomic_bonds + 1
                else:
                    n_homogenous_bonds += 1

        return n_homogenous_bonds, n_heteroatomic_bonds

    def get_atom_indices_from_coordination_number(self, coordination_numbers, symbol=None):
        if symbol is None:
            return list(filter(lambda x: self.get_coordination_number(x) in coordination_numbers, self.atoms.get_indices()))
        else:
            return list(filter(lambda x: self.get_coordination_number(x) in coordination_numbers and self.atoms.get_symbol(x) == symbol, self.atoms.get_indices()))

    def get_coordination_number(self, lattice_index):
        return self.neighbor_list.getCoordinationNumber(lattice_index)

    def get_atoms(self, atomIndices=None):
        return copy.deepcopy(self.atoms.get_atoms(atomIndices))

    def get_neighbor_list(self):
        return self.neighbor_list

    def get_ASE_atoms(self, centered=True):
        atom_positions = list()
        atomic_symbols = list()
        for lattice_index in self.atoms.get_indices():
            atom_positions.append(self.lattice.getCartesianPositionFromIndex(lattice_index))
            atomic_symbols.append(self.atoms.get_symbol(lattice_index))

        atoms = Atoms(positions=atom_positions, symbols=atomic_symbols)
        if centered:
            COM = atoms.get_center_of_mass()
            return Atoms(positions=[position - COM for position in atom_positions], symbols=atomic_symbols)
        else:
            return Atoms(positions=atom_positions, symbols=atomic_symbols)

    def get_potential_energy(self, steps=100):
        cell_width = self.lattice.width*self.lattice.latticeConstant
        cell_length = self.lattice.length*self.lattice.latticeConstant
        cell_height = self.lattice.height*self.lattice.latticeConstant

        atoms = self.get_ASE_atoms()
        atoms.set_cell(np.array([[cell_width, 0, 0], [0, cell_length, 0], [0, 0, cell_height]]))
        atoms.set_calculator(EMT())
        dyn = BFGS(atoms)
        dyn.run(fmax=0.01, steps=steps)

        return atoms.get_potential_energy()

    def get_stoichiometry(self):
        return self.atoms.get_stoichiometry()

    def get_surface_vacancies(self):
        not_fully_coordinated_atoms = self.get_atom_indices_from_coordination_number(range(self.lattice.MAX_NEIGHBORS))
        surface_vacancies = set()

        for atom in not_fully_coordinated_atoms:
            neighbor_vacancies = self.lattice.getNearestNeighbors(atom).difference(self.neighbor_list[atom])
            surface_vacancies = surface_vacancies.union(neighbor_vacancies)
        return surface_vacancies

    def get_atomic_neighbors(self, index):
        neighbors = list()
        nearest_neighbors = self.lattice.getNearestNeighbors(index)
        for lattice_index in nearest_neighbors:
            if lattice_index in self.atoms.get_indices():
                neighbors.append(lattice_index)

        return neighbors

    def get_n_atomic_neighbors(self, index):
        return len(self.get_atomic_neighbors(index))

