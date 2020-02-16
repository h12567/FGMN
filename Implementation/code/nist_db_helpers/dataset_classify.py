import os
import json
from rdkit import Chem
import tqdm
from collections import defaultdict
import util

def count_func_group_from_smart():
    cwd = os.path.dirname(__file__)
    path_smart = os.path.join(os.path.dirname(cwd), "../data/smarts.json")
    path_dir_molecule = os.path.join(os.path.dirname(cwd), '../data/nist_database', 'mol')
    fnames_molecule = [fn for fn in os.listdir(path_dir_molecule) if fn.endswith('mol')]
    fnames_molecule = [os.path.splitext(fn)[0] for fn in fnames_molecule]  # remove ext
    func_group_cnt = defaultdict(lambda: 0)
    with open(path_smart) as json_file:
        func_group_mapping = json.load(json_file)
        for fname in tqdm.tqdm(fnames_molecule):
            filepath = os.path.join(path_dir_molecule, fname + '.mol')
            molecule = Chem.SDMolSupplier(filepath)[0]
            if molecule is None:
                continue  # Some .mol files that actually have no molecule in them. Crawler issue?
            try:
                _, smile, _ = util.extract_structure(filepath)  # Some .mol files cannot be loaded.
                mol = Chem.RWMol(Chem.MolFromSmiles(smile))
                for func_group, smart_rep in func_group_mapping.items():
                    smart_rep_group = Chem.MolFromSmarts(smart_rep)
                    matches = mol.GetSubstructMatches(smart_rep_group)
                    if len(matches) > 0:
                        func_group_cnt[func_group] += 1
            except Exception as e:
                print(e)
    return func_group_cnt

def count_max_and_unique_atoms_from_smart(
        func_group=None, allow_molecules=None, max_constraint=None,
):
    atom_type_set = set()
    max_atoms = 0
    cwd = os.path.dirname(__file__)
    path_smart = os.path.join(os.path.dirname(cwd), "../data/smarts.json")
    path_dir_molecule = os.path.join(os.path.dirname(cwd), '../data/nist_database', 'mol')
    fnames_molecule = [fn for fn in os.listdir(path_dir_molecule) if fn.endswith('mol')]
    fnames_molecule = [os.path.splitext(fn)[0] for fn in fnames_molecule]  # remove ext
    with open(path_smart) as json_file:
        func_group_mapping = json.load(json_file)
        for fname in tqdm.tqdm(fnames_molecule):
            filepath = os.path.join(path_dir_molecule, fname + '.mol')
            molecule = Chem.SDMolSupplier(filepath)[0]
            if molecule is None:
                continue  # Some .mol files that actually have no molecule in them. Crawler issue?
            try:
                _, smile, _ = util.extract_structure(filepath)  # Some .mol files cannot be loaded.
                mol = Chem.RWMol(Chem.MolFromSmiles(smile))
                smart_rep = func_group_mapping[func_group]
                smart_rep_group = Chem.MolFromSmarts(smart_rep)
                matches = mol.GetSubstructMatches(smart_rep_group)
                if len(matches) > 0:
                    max_atoms = max(max_atoms, len(mol.GetAtoms()))
                    for a in mol.GetAtoms():
                        atom_symbol = a.GetSymbol()
                        atom_type_set.add(atom_symbol)
            except Exception as e:
                print(e)
    return atom_type_set, max_atoms

# func_group_cnt = count_func_group_from_smart()
atoms_set, max_atoms = count_max_and_unique_atoms_from_smart(
    func_group="ester", allow_molecules=("C", "H", "O", "N", "P", "S"), max_constraint=("C", 5),
)
a = 1
