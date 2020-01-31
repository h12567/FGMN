from rdkit import Chem
import util
import copy
import os
import tqdm
import numpy as np
import ast

def print_atoms(molecule):
    for atom in molecule.GetAtoms():
        symbol=atom.GetSymbol()
        formal_charge=atom.GetFormalCharge()
        implicit_valence=atom.GetImplicitValence()
        ring_atom=atom.IsInRing()
        degree=atom.GetDegree()
        hybridization=atom.GetHybridization()
        radicals = atom.GetNumRadicalElectrons()
        isaromatic = atom.GetIsAromatic()
        print(symbol, formal_charge, implicit_valence, ring_atom, degree, hybridization, radicals, isaromatic)

def stats_smiles(smile):
    # returns num bonds and num atoms in cases of including and excluding Hs.
    stats = {}
    m = Chem.MolFromSmiles(smile)
    m = Chem.RemoveHs(m)
    stats['nbonds_no_H'] = m.GetNumBonds()
    stats['natoms_no_H'] = m.GetNumHeavyAtoms()
    m = Chem.AddHs(m)
    stats['nbonds_include_H'] = m.GetNumBonds()
    stats['natoms_include_H'] = m.GetNumAtoms()
    return stats

def has_square_brackets(smile):
    idx1 = smile.find('[')
    idx2 = smile.find(']')
    return idx1 != -1 or idx2 != -1

def H_atoms_in_molecule(molecule):
    # example: H_atoms_in_molecule(Chem.MolFromSmiles('CCO')) returns 6.
    m2 = Chem.AddHs(molecule) #deepcopied molecule object.
    return m2.GetNumAtoms() - molecule.GetNumAtoms()

def similarity_fingerprint_default(molecule1, molecule2):
    # example in example_molecule_similarity.py
    fp1 = FingerprintMols.FingerprintMol(molecule1)
    fp2 = FingerprintMols.FingerprintMol(molecule2)
    return DataStructs.FingerprintSimilarity(fp1, fp2) # default is tanimoto, can change to dice.

def similarity_morgan_dice(molecule1, molecule2, atom_radius):
    # example in example_molecule_similarity.py
    FastFindRings(molecule1)
    FastFindRings(molecule2)
    fp1 = AllChem.GetMorganFingerprint(molecule1, atom_radius)
    fp2 = AllChem.GetMorganFingerprint(molecule2, atom_radius)
    return DataStructs.DiceSimilarity(fp1, fp2)

# TODO(Bowen): check, esp if input is not radical
def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def get_useable_nist_data(force_recompute=False, possible_atoms=None, min_n_atoms=0, limit_n_atoms=999999, ban_square_brackets_smiles=False, ban_rare_atomic_mass=True, ban_charges=True, ban_2H=True, ban_wildcard=True, whole_molecule_0_implicit_valence=False, path_save_info=None, print_smiles=False, dataset='nist', filenames_care_about=None):
    '''
    Example call of this function is in get_nist_data_subset.py . You can see how each argument is set.
    '''

    import re

    cwd = os.path.dirname(__file__)
    if 'nist' == dataset:
        path_save_info = path_save_info if path_save_info else os.path.join(cwd, 'useable_nist_data.txt')
        path_dir_mass_spectrum = os.path.join(os.path.dirname(cwd), '../data/nist_database', 'jdx')
        path_dir_molecule = os.path.join(os.path.dirname(cwd), '../data/nist_database', 'mol')
        print('David: expect 52075 number mass spectrum files (.jdx) and molecule fiels (.mol).')
    elif 'opcw_phosphonotionates' == dataset:
        path_save_info = path_save_info if path_save_info else os.path.join(cwd, 'useable_opcw_phosphonotionates_data.txt')
        path_dir_mass_spectrum = os.path.join(os.path.dirname(cwd), 'dataset', 'opcw_database', 'jdx')
        path_dir_molecule = os.path.join(os.path.dirname(cwd), 'dataset', 'opcw_database', 'mol')
    else:
        raise Exception('no such option')

    if not os.path.exists(path_save_info) or force_recompute: # Compute useable daya and write into useable_nist_data.txt
        ### Retrieve filenames appearing in both mol & jdx directories:
        fnames_mass_spectrum = [fn for fn in os.listdir(path_dir_mass_spectrum) if fn.endswith('jdx')]
        fnames_molecule = [fn for fn in os.listdir(path_dir_molecule) if fn.endswith('mol')]
        print(f'Counted {len(fnames_mass_spectrum)} number mass spectrum files (.jdx) in {path_dir_mass_spectrum} BEFORE filters applied.')
        print(f'Counted {len(fnames_molecule)} number molecule files (.mol) in {path_dir_molecule} BEFORE filters applied.')
        # print(fnames_mass_spectrum[0])
        fnames_mass_spectrum = [os.path.splitext(fn)[0] for fn in fnames_mass_spectrum] # remove ext
        fnames_molecule = [os.path.splitext(fn)[0] for fn in fnames_molecule] # remove ext
        fnames = list(set(fnames_mass_spectrum).intersection(fnames_molecule))
        print('Num files in both directories', len(fnames))
        ### Remove problematic molecule files.
        ### Also, count max number of atoms.
        max_n_atoms = -1
        max_n_bonds_include_H = -10
        min_n_bonds_include_H = 1e9
        max_n_bonds_no_H = -10
        min_n_bonds_no_H = 1e9
        print('Identifying non-problematic .mol files please wait...')
        fnames_old = copy.copy(fnames)
        fnames = []
        for fname in tqdm.tqdm(fnames_old):
            if filenames_care_about is not None:
                if fname not in filenames_care_about:
                    continue #skip
            try:
                filepath = os.path.join(path_dir_molecule, fname + '.mol')
                molecule = Chem.SDMolSupplier(filepath)[0]
                if molecule is None:
                    continue # Some .mol files that actually have no molecule in them. Crawler issue?
                try:
                    _, smile, _ = util.extract_structure(filepath) # Some .mol files cannot be loaded.
                    mol = Chem.RWMol(Chem.MolFromSmiles(smile))
                    stats = stats_smiles(smile)
                    atoms = mol.GetAtoms()
                    n_atoms = stats['natoms_no_H']
                    if n_atoms > limit_n_atoms:
                        continue #skip this molecule
                    if n_atoms < min_n_atoms:
                        continue #skip this molecule
                    if possible_atoms:
                        impossible_atoms_found = [str(atom.GetSymbol()) for atom in atoms if str(atom.GetSymbol()) not in possible_atoms]
                        if len(impossible_atoms_found) > 0:
                            continue #skip this molecule
                    for atom in atoms:
                        if whole_molecule_0_implicit_valence:
                            if atom.GetImplicitValence() > 0:
                                continue #skip this molecule
                        if ban_charges:
                            if atom.GetFormalCharge() > 0 or atom.GetNumRadicalElectrons() > 0:
                                continue #skip this molecule
                    if ban_square_brackets_smiles:
                        if has_square_brackets(smile):
                            continue #skip this molecule
                    if ban_rare_atomic_mass:
                        match = re.search('\[[0-9]+[a-zA-Z]+[@0-9a-zA-Z]*\]', smile) # catches [13C] [2H] [13CH] [13C@H] [13C@@H] [13C987XY]
                        if match:
                            continue #skip this molecule
                    if ban_2H:
                        if -1 != smile.find('[2H]') or -1 != smile.find('[3H]') or -1 != smile.find('[4H]'):
                            continue #skip this molecule
                    if ban_wildcard:
                        if -1 != smile.find('*'):
                            continue #skip this molecule
                    # Add to running stats after passing checks:
                    max_n_atoms = max(n_atoms, max_n_atoms)
                    max_n_bonds_include_H = max(stats['nbonds_include_H'], max_n_bonds_include_H)
                    max_n_bonds_no_H = max(stats['nbonds_no_H'], max_n_bonds_no_H)
                    min_n_bonds_include_H = min(stats['nbonds_include_H'], min_n_bonds_include_H)
                    min_n_bonds_no_H = min(stats['nbonds_no_H'], min_n_bonds_no_H)
                    fnames.append(fname)
                    if print_smiles:
                        print(len(fnames)-1, smile)
                except:
                    continue
            except Exception as e:
                print(fname, e)
        print('Num files after removing problematic .mol files', len(fnames))
        del fnames_old
        ### Deduce length of mass spectrum x axis:
        print('Deducing length of mass spectrum x axis please wait ...')
        len_mass_spectrum_x_axis = -1
        for fname in tqdm.tqdm(fnames):
            *_, spikes = util.read_mass_spec(os.path.join(path_dir_mass_spectrum, fname+'.jdx'), x_axis=5000)
            spikes = np.array(spikes)
            # print(spikes, spikes.shape)
            w = np.where(spikes > 0)[0]
            this_x_axis = max(w) + 1 # length of x-axis needed to hold all data of this mass spectrum.
            len_mass_spectrum_x_axis = max(len_mass_spectrum_x_axis, this_x_axis)
        print('Deduced length of mass spectrum x axis:', len_mass_spectrum_x_axis)
        ### Write to file
        info_write = {}
        info_write['fnames'] = fnames
        info_write['n_fnames'] = len(fnames)
        info_write['len_mass_spectrum_x_axis'] = len_mass_spectrum_x_axis
        info_write['min_n_atoms'] = min_n_atoms
        info_write['max_n_atoms'] = max_n_atoms
        info_write['limit_n_atoms'] = limit_n_atoms
        info_write['possible_atoms'] = possible_atoms
        info_write['whole_molecule_0_implicit_valence'] = whole_molecule_0_implicit_valence
        info_write['max_n_bonds_include_H'] = max_n_bonds_include_H
        info_write['min_n_bonds_include_H'] = min_n_bonds_include_H
        info_write['max_n_bonds_no_H'] = max_n_bonds_no_H
        info_write['min_n_bonds_no_H'] = min_n_bonds_no_H
        info_write['ban_square_brackets_smiles'] = ban_square_brackets_smiles
        info_write['ban_rare_atomic_mass'] = ban_rare_atomic_mass
        info_write['ban_2H'] = ban_2H
        info_write['ban_charges'] = ban_charges
        info_write['ban_wildcard'] = ban_wildcard
        write_file_data_subset(path_save_info, info_write)
    else: ### Do no compute. Just load previously saved info in text file `path_save_info`.
        pass
    return read_file_data_subset(path_save_info)



def write_file_data_subset(path_save_info, info):
    with open(path_save_info, 'w') as f:
        f.write("##useable={}".format(info['fnames']))
        f.write('\n')
        f.write("##n_useable={}".format(info['n_fnames']))
        f.write('\n')
        f.write("##len_mass_spectrum_x_axis={}".format(info['len_mass_spectrum_x_axis']))
        f.write('\n')
        f.write("##min_n_atoms={}".format(info['min_n_atoms']))
        f.write('\n')
        f.write("##max_n_atoms={}".format(info['max_n_atoms']))
        f.write('\n')
        f.write("##limit_n_atoms={}".format(info['limit_n_atoms']))
        f.write('\n')
        f.write("##possible_atoms={}".format(info['possible_atoms']))
        f.write('\n')
        f.write("##whole_molecule_0_implicit_valence={}".format(info['whole_molecule_0_implicit_valence']))
        f.write('\n')
        f.write("##max_n_bonds_include_H={}".format(info['max_n_bonds_include_H']))
        f.write('\n')
        f.write("##min_n_bonds_include_H={}".format(info['min_n_bonds_include_H']))
        f.write('\n')
        f.write("##max_n_bonds_no_H={}".format(info['max_n_bonds_no_H']))
        f.write('\n')
        f.write("##min_n_bonds_no_H={}".format(info['min_n_bonds_no_H']))
        f.write('\n')
        f.write("##ban_square_brackets_smiles={}".format(info['ban_square_brackets_smiles']))
        f.write('\n')
        f.write("##ban_rare_atomic_mass={}".format(info['ban_rare_atomic_mass']))
        f.write('\n')
        f.write("##ban_2H={}".format(info['ban_2H']))
        f.write('\n')
        f.write("##ban_charges={}".format(info['ban_charges']))
        f.write('\n')
        f.write("##ban_wildcard={}".format(info['ban_wildcard']))

def read_file_data_subset(path_save_info):
    with open(path_save_info, 'r') as f:
        more_info = {}
        for line in f:
            if line.startswith('##len_mass_spectrum_x_axis'):
                len_mass_spectrum_x_axis = int(line.rstrip().split('=')[1])
            elif line.startswith('##useable'):
                liststr = str(line.rstrip().split('=')[1])
                # print(liststr[:100])
                fnames = ast.literal_eval(liststr)
            elif line.startswith('##n_useable'):
                more_info['n_useable'] = int(line.rstrip().split('=')[1])
            elif line.startswith('##min_n_atoms'):
                more_info['min_n_atoms'] = int(line.rstrip().split('=')[1])
            elif line.startswith('##max_n_atoms'):
                more_info['max_n_atoms'] = int(line.rstrip().split('=')[1])
            elif line.startswith('##limit_n_atoms'):
                more_info['limit_n_atoms'] = int(line.rstrip().split('=')[1])
            elif line.startswith('##possible_atoms'):
                liststr = str(line.rstrip().split('=')[1])
                more_info['possible_atoms'] = ast.literal_eval(liststr)
            elif line.startswith('##whole_molecule_0_implicit_valence'):
                recordstr = str(line.rstrip().split('=')[1])
                more_info['whole_molecule_0_implicit_valence'] = ast.literal_eval(recordstr)
            elif line.startswith('##max_n_bonds_include_H'):
                more_info['max_n_bonds_include_H'] = int(line.rstrip().split('=')[1])
            elif line.startswith('##min_n_bonds_include_H'):
                more_info['min_n_bonds_include_H'] = int(line.rstrip().split('=')[1])
            elif line.startswith('##max_n_bonds_no_H'):
                more_info['max_n_bonds_no_H'] = int(line.rstrip().split('=')[1])
            elif line.startswith('##min_n_bonds_no_H'):
                more_info['min_n_bonds_no_H'] = int(line.rstrip().split('=')[1])
            elif line.startswith('##ban_square_brackets_smiles'):
                more_info['ban_square_brackets_smiles'] = ast.literal_eval(line.rstrip().split('=')[1])
            elif line.startswith('##ban_rare_atomic_mass'):
                more_info['ban_rare_atomic_mass'] = ast.literal_eval(line.rstrip().split('=')[1])
            elif line.startswith('##ban_2H'):
                more_info['ban_2h'] = ast.literal_eval(line.rstrip().split('=')[1])
            elif line.startswith('##ban_charges'):
                more_info['ban_charges'] = ast.literal_eval(line.rstrip().split('=')[1])
            elif line.startswith('##ban_wildcard'):
                more_info['ban_wildcard'] = ast.literal_eval(line.rstrip().split('=')[1])
    return fnames, len_mass_spectrum_x_axis, more_info
