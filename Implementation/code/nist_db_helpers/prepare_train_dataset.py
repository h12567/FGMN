import os
import json
from rdkit import Chem
import tqdm
from collections import defaultdict
import numpy as np
from rdkit.Chem.rdchem import BondType
import util

d_n = 800

def is_valid_mol(
        mol=None, func_group=None, allow_molecules=None, max_constraint=None,
):
    cwd = os.path.dirname(__file__)
    path_smart = os.path.join(os.path.dirname(cwd), "../data/smarts.json")
    with open(path_smart) as json_file:
        # Validate functional group
        func_group_mapping = json.load(json_file)
        smart_rep = func_group_mapping[func_group]
        smart_rep_group = Chem.MolFromSmarts(smart_rep)
        matches = mol.GetSubstructMatches(smart_rep_group)
        if len(matches) == 0:
            return False

        # validate allow molecules
        for a in mol.GetAtoms():
            atom_symbol = a.GetSymbol()
            if atom_symbol not in allow_molecules:
                return False

        # validate max constraint
        mol_freq = {}
        for a in mol.GetAtoms():
            atom_symbol = a.GetSymbol()
            if atom_symbol in mol_freq:
                mol_freq[atom_symbol] += 1
            else:
                mol_freq[atom_symbol] = 1

        for atom, freq in max_constraint:
            if atom in mol_freq and mol_freq[atom] > freq:
                return False

        return True
    return False

def generate_mols_msp():
    cwd = os.path.dirname(__file__)
    path_smart = os.path.join(os.path.dirname(cwd), "../data/smarts.json")
    path_dir_mass_spectrum = os.path.join(os.path.dirname(cwd), '../data/nist_database', 'jdx')
    path_dir_molecule = os.path.join(os.path.dirname(cwd), '../data/nist_database', 'mol')
    fnames_mass_spectrum = [fn for fn in os.listdir(path_dir_mass_spectrum) if fn.endswith('jdx')]
    fnames_mass_spectrum = [os.path.splitext(fn)[0] for fn in fnames_mass_spectrum]  # remove ext
    fnames_molecule = [fn for fn in os.listdir(path_dir_molecule) if fn.endswith('mol')]
    fnames_molecule = [os.path.splitext(fn)[0] for fn in fnames_molecule]  # remove ext
    fnames = list(set(fnames_mass_spectrum).intersection(fnames_molecule))
    len_mass_spectrum_x_axis = -1

    with open(path_smart) as json_file:
        for fname in tqdm.tqdm(fnames):
            mol_filepath = os.path.join(path_dir_molecule, fname + '.mol')
            molecule = Chem.SDMolSupplier(mol_filepath)[0]
            if molecule is None:
                continue  # Some .mol files that actually have no molecule in them. Crawler issue?
            msp_filepath = os.path.join(path_dir_mass_spectrum, fname+'.jdx')
            try:
                x, y, spikes = util.read_mass_spec(msp_filepath, x_axis=d_n)
                spikes = np.array(spikes)
                _, smile, _ = util.extract_structure(mol_filepath)  # Some .mol files cannot be loaded.
                mol = Chem.RWMol(Chem.MolFromSmiles(smile))
                yield mol, spikes
            except Exception as e:
                print(e)

def count_max_and_unique_atoms_from_smart(
        func_group=None, allow_molecules=None, max_constraint=None,
):
    atom_type_set = set()
    max_atoms = 0
    for mol, _ in generate_mols_msp():
        if is_valid_mol(
            mol=mol, func_group=func_group, allow_molecules=allow_molecules, max_constraint=max_constraint,
        ):
            max_atoms = max(max_atoms, len(mol.GetAtoms()))
            for a in mol.GetAtoms():
                atom_symbol = a.GetSymbol()
                atom_type_set.add(atom_symbol)
    return atom_type_set, max_atoms

def extract_vertex_idxes(mol, allow_molecules):
    vertex_arr = []
    # TODO: Get canonical order instead of just mol.GetAtoms()
    for a in mol.GetAtoms():
        atom_symbol = a.GetSymbol()
        occur_idx = [i for i in range(len(allow_molecules)) if allow_molecules[i] == atom_symbol][0]
        vertex_arr.append(occur_idx)
    return vertex_arr

def extract_adj_matrix(mol, possible_bonds, max_atoms):
    E = np.zeros((len(possible_bonds) + 1, max_atoms, max_atoms))
    for b in mol.GetBonds():
        begin_idx = b.GetBeginAtomIdx()
        end_idx = b.GetEndAtomIdx()
        bond_type = b.GetBondType()
        float_array = (bond_type == np.array(possible_bonds)).astype(float)
        E[:, begin_idx, end_idx] = np.insert(float_array, 0, 0)
        E[:, end_idx, begin_idx] = np.insert(float_array, 0, 0)
    return E

def prepare_training(
        func_group=None, allow_molecules=None, max_constraint=None,
        possible_bonds=None, max_atoms=None,
):
    all_mol_vertex_arr = []
    all_mol_adj_arr = []
    all_msp_arr = []
    max_spike = -1
    for mol, spikes in generate_mols_msp():
        if is_valid_mol(
                mol=mol, func_group=func_group, allow_molecules=allow_molecules,
                max_constraint=max_constraint,
        ):
            vertex_arr = extract_vertex_idxes(mol, allow_molecules)
            max_spike = max(max(spikes), max_spike)
            E = extract_adj_matrix(mol, possible_bonds, max_atoms)
            all_mol_vertex_arr.append(vertex_arr)
            all_msp_arr.append(spikes)
            all_mol_adj_arr.append(E)
    np.save("vertex_arr.npy", all_mol_vertex_arr)
    np.save("mol_adj_arr.npy", all_mol_adj_arr)
    np.save("msp_arr.npy", all_msp_arr)

func_group = "ester"
allow_molecules = ["C", "H", "O", "N", "P", "S"]
max_constraint = [("C", 5)]
possible_bonds = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]
max_atoms = 13
# atom_type_set, max_atoms = count_max_and_unique_atoms_from_smart(
#     func_group=func_group, allow_molecules=allow_molecules, max_constraint=max_constraint,
# )
# atom_type_set: ("N", "C", "O", "S", "P")
# max_atoms: 13
prepare_training(
    func_group=func_group, allow_molecules=allow_molecules, max_constraint=max_constraint,
    possible_bonds=possible_bonds, max_atoms=max_atoms,
)
a = 1
