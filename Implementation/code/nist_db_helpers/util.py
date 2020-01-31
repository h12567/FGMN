from collections import defaultdict
# from .opensmilesLexer import opensmilesLexer
from rdkit import Chem
from scipy.sparse import coo_matrix

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

import antlr4
import igraph as ig
import json
import jsonpickle
import numpy as np
from collections import Counter as Multiset

acceptable_atoms = {'C', 'O', 'P', 'N', 'S', 'F', 'Si', 'Cl', 'H', 'Br', 'As', 'I', 'B'}

class Structure:
    def __init__(self, mass, smi, graph):
        tokens, spaced = process_smi(smi)
        self.mass = mass
        self.smi = smi
        self.smi_tokens = tokens
        self.spaced_smi = spaced
        self.graph = graph
        self.io_frags = None


class Compound:
    def __init__(self, name, molform, spikes, num_candidates, candidates):
        self.name = name
        self.molform = molform
        self.spikes = spikes
        #self.reference_id = reference_id
        self.num_candidates = num_candidates
        self.candidates = candidates


def get_indices(idx_file, dtype=int):
    """ LOADS INDICES FROM FILE
        loaded as int by default
        idx_file: one index per line """
    return list(map(dtype, open(idx_file).read().split()))


def read_json(fname):
    """ Reads obj from JSON file """
    return jsonpickle.decode(json.load(open(fname, 'r')))


def write_json(obj, fname):
    """ Writes obj into fname in JSON format """
    json.dump(jsonpickle.encode(obj), open(fname, 'w'), indent=4)


###################
# NN MODEL CONFIG #
###################

class Config:
    """ CONFIG INITIALIZER
        Use this to change config EXCEPT MAX_SEQUENCE_LEN, VOCAB_SIZE and molCount_len.
        They are set automatically during training.
        Saved as 'config.json' (when training). """

    def __init__(self):
        # FOR TRAIN
        self.ngpus = 1  # auto set during training
        self.epochs = 100
        self.batch_size = 64
        self.test_batch_size = 10
        
        # MODEL
        self.nOutput = 0  # auto set during training
        self.nFilters = 100
        self.filter_sizes = [3, 4, 5, 6, 7]
        self.filter_sizes2 = [2, 3, 4]
        
        self.ms_upper = 500
        self.molCount_len = 0  # auto set during training
        

############################
# HANDLING MOLFORM STRINGS #
############################

def get_char_type(char):
    if char.isdigit():
        return 'DIGIT'
    elif char.islower():
        return 'LOWER'
    elif char.isupper():
        return 'UPPER'
    else:
        return None


def get_base_type(char):
    if char.isdigit():
        return int
    elif char.isalpha():
        return str


def split_alphanum(alphanum, keep_as_string=False):
    new_alphanum = alphanum[0]
    base_dtype = type(alphanum[0])
    for a in alphanum[1:]:
        btype = get_base_type(a) #base type
        dtype = get_char_type(a) #data type
        if btype == base_dtype:
            if dtype == 'UPPER':
                new_alphanum += ' ' + a
            else:
                new_alphanum += a
        elif btype != base_dtype: 
            new_alphanum += ' ' + a
            base_dtype = btype
    
    if keep_as_string:
        return new_alphanum.rstrip()
    else:
        return new_alphanum.rstrip().split(' ')


def prep_alphanum(alphanum):
    """ alphanum: list of split alphanum """
    new_alphanum = alphanum
    
    if alphanum[-1].isdigit():
        return new_alphanum
    elif alphanum[-1].isalpha():
        new_alphanum.append('1')
        return new_alphanum
    else:
        print("Error preparing alphanum. Last item not digit or alpha.")
        print("alphanum", alphanum)
        exit()


def get_prep_alphanum(alphanum):
    """ split and prepare alphanum """
    return prep_alphanum(split_alphanum(alphanum))


def get_class_count(data_list):
    nClasses = len(data_list[0])
    counts = [0] * nClasses
    for data in data_list:
        for i, d in enumerate(data):
            if d == 1:
                counts[i] += 1
    
    with open('train_cnt.txt', 'w') as f:
        for i in range(nClasses):
            f.write("\n".join(map(str, counts)))


def acceptable_condition(filename):
    """ given a file, check if all atoms are acceptable """

    lines = open(filename).readlines()
    lines = lines[4:4+int(lines[3].split()[0])]  # extract lines with atoms
    atoms = {line.split()[3] for line in lines}  # extract atoms
    if atoms - acceptable_atoms:
        print(filename, "is in unacceptable condition!")
        return False
    return True


def process_smi(s):
    lexer = opensmilesLexer(antlr4.InputStream(s))
    tokens = []
    while True:
        token = lexer.nextToken().text
        if token == '<EOF>':
            break
        elif token.isalpha() and token not in acceptable_atoms:
            tokens.extend(list(token))  # splits up the token into smaller bits
        else:
            tokens.append(token)
    spaced = ''
    for i in range(len(tokens)):
        spaced += tokens[i]
        if i != len(tokens) - 1:
            spaced += ' '
    return tokens, spaced


def read_mass_spec(fname, x_axis=800):
    """ Given a the path to a jdx file (and possibly x_axis range),
        returns 1) name of the molecule
                2) molecular formula of the molecule
                3) mass spec spikes in the form of spikes[x] = y-value at m/z = x,
                   for x in given range (default = 800)
        Note: 4933.jdx has m/z at 529, and 5613.jdx has m/z at 718 """
    name = None
    molform = None
    spikes = [0] * x_axis
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith('##CAS NAME'):
                name = line.rstrip().split('=')[1].upper()
            elif line.startswith('##MOLFORM'):
                molform = line.rstrip().split('=')[1]
            elif line.startswith('##XYDATA'):
                for line in f:
                    line = line.split()
                    spikes[int(line[0])] = float(line[1])
    return name, molform, spikes


def extract_structure(fname):
    H_mass=1.008

    molecule = Chem.SDMolSupplier(fname)[0]
    if molecule is None:
        print("Error in parsing from {}".format(fname))
        exit()

    graph = ig_create(Chem.MolToMolBlock(molecule).split('\n'))
    smi = Chem.MolToSmiles(molecule)
    mass = 0
    for atom in molecule.GetAtoms():
        mass += atom.GetMass() + atom.GetImplicitValence() * H_mass

    return mass, smi, graph


def extract_structures(fname):
    """ Given the path to a file, return a list of Structure objects parsed from fname """
    H_mass = 1.008
    structures = []
    mol_supplier = Chem.SDMolSupplier(fname)
    for i, molecule in enumerate(mol_supplier):
        if molecule == None:
            print("Error in parsing {0}-th structure (1-indexed) from {1}".format(i+1, fname))
            #print(mol_supplier.GetItemText(i))
            continue
        graph = ig_create(Chem.MolToMolBlock(molecule).split("\n"))
        smi = Chem.MolToSmiles(molecule)
        mass = 0
        for atom in molecule.GetAtoms():
            mass += atom.GetMass() + atom.GetImplicitValence()*H_mass
        structures.append(Structure(mass, smi, graph))
    return structures


def locate_reference(candidate_graphs, reference_graph):
    """ Given a list of candidate graphs and the reference graph,
        returns the index of the reference graph within the candidate list """
    reference_idx = None
    g = reference_graph
    for idx in range(len(candidate_graphs)):
        cg = candidate_graphs[idx]
        if ig_isomorphic(g, cg):
            assert(reference_idx == None)
            reference_idx = idx

    assert(reference_idx != None)
    return reference_idx


###############
# GRAPH STUFF #
###############

def ig_create(molblock):
    params = molblock[3].split()
    n = int(params[0])
    m = int(params[1])
    lines = molblock[4:]
    g = ig.Graph()
    g.add_vertices(n)
    for i in range(n):
        vertex_line = lines[i].split()
        atom = vertex_line[3]
        g.vs[i]["atom"] = atom
    for i in range(m):
        edge_line = lines[n+i].split()
        u = int(edge_line[0]) - 1
        v = int(edge_line[1]) - 1
        w = int(edge_line[2])
        g.add_edges([(u,v)])
        g.es[g.get_eid(u,v)]["weight"] = w
    return g


def ig_extract_attributes(g1, g2):
    atom_map = dict()
    for atom in g1.vs["atom"]:
        if atom not in atom_map.keys():
            atom_map[atom] = len(atom_map) + 1
    for atom in g2.vs["atom"]:
        if atom not in atom_map.keys():
            atom_map[atom] = len(atom_map) + 1
    atoms_g1 = [atom_map[atom] for atom in g1.vs["atom"]] if g1.vcount() >= 1 else []
    atoms_g2 = [atom_map[atom] for atom in g2.vs["atom"]] if g2.vcount() >= 1 else []
    weights_g1 = g1.es["weight"] if g1.ecount() >= 1 else []
    weights_g2 = g2.es["weight"] if g2.ecount() >= 1 else []
    return atoms_g1, atoms_g2, weights_g1, weights_g2


def ig_isomorphic(g1, g2):
    atoms_g1, atoms_g2, weights_g1, weights_g2 = ig_extract_attributes(g1, g2)
    # Somehow isomorphic_vf2 assumes g1 and g2 have same number of |V| and |E|?!
    # It crashes without this check... How strange
    if g1.ecount() != g2.ecount() or g1.vcount() != g2.vcount():
        return False
    return g1.isomorphic_vf2(g2,
                             color1=atoms_g1,
                             color2=atoms_g2,
                             edge_color1=weights_g1,
                             edge_color2=weights_g2)


def ig_subisomorphic(sg, g):
    atoms_sg, atoms_g, weights_sg, weights_g = ig_extract_attributes(sg, g)
    return g.subisomorphic_vf2(sg,
                               color1=atoms_g,
                               color2=atoms_sg,
                               edge_color1=weights_g,
                               edge_color2=weights_sg)


def preprocess(graphs):
    preprocessed = []
    for g in graphs:
        sorted_atoms = []
        sorted_adjlist = defaultdict(list)
        for i in range(g.vcount()):
            if len(g.get_adjlist()[i]): # Some fragments have lone disjoint atoms
                atom = g.vs["atom"][i]
                sorted_atoms.append(atom)
                edges = sorted(g.vs['atom'][j] for j in g.get_adjlist()[i])
                sorted_adjlist[atom].append(edges)
        preprocessed.append((sorted(sorted_atoms), sorted_adjlist))
    return preprocessed


def subset(v1, v2):
    if len(v1) > len(v2):
        return False
    i = j = 0
    while i < len(v1):
        if j == len(v2):
            return False
        elif v1[i] == v2[j]:
            i += 1
        j += 1
    return True


# To consider: Use 2 decimal places to count up to 99 instances of an atom, then hash neighbours into a number
# e.g. 2C, 1O => 0201 or something.
def ssubset(vv1, vv2):
    assert(len(vv1) <= len(vv2))
    for i in range(len(vv1)):
        for j in range(len(vv2)):
            if subset(vv1[i], vv2[j]):
                break
        else:
            return False
    return True


def davin_subisomorphic(subgraphs, graphs):
    m = len(graphs)
    n = len(subgraphs)
    print("Building io_matrix of dimension {} x {}".format(m, n))
    io_matrix = np.zeros((m, n)).astype(int)

    # Pre-process all subgraphs and graphs
    pre_sg = preprocess(subgraphs)
    pre_g  = preprocess(graphs)

    # Perform subgraph isomorphism check
    for i, g in enumerate(graphs):
        atoms_g, adjlist_g = pre_g[i]
        for j, sg in enumerate(subgraphs):
            atoms_sg, adjlist_sg = pre_sg[j]
            if subset(atoms_sg, atoms_g):
                for atom in adjlist_sg.keys():
                    list_sg = adjlist_sg[atom]
                    list_g = adjlist_g[atom]
                    if not ssubset(list_sg, list_g):
                        break
                else:
                    if ig_subisomorphic(sg, g):
                        io_matrix[i, j] = 1
    return io_matrix


def sparse_subisomorphic(subgraphs, graphs):
    m = len(graphs)
    n = len(subgraphs)
    print('Building sparse io_matrix of dimension {} x {}'.format(m, n))

    pre_sg = preprocess(subgraphs)

    row, col = [], []
    for i, g in enumerate(graphs):
        atoms_g, adjlist_g = preprocess([g])[0]
        for j, sg in enumerate(subgraphs):
            atoms_sg, adjlist_sg = pre_sg[j]
            if subset(atoms_sg, atoms_g):
                for atom in adjlist_sg.keys():
                    list_sg = adjlist_sg[atom]
                    list_g = adjlist_g[atom]
                    if not ssubset(list_sg, list_g):
                        break
                else:
                    if ig_subisomorphic(sg, g):
                        row.append(i)
                        col.append(j)
    return coo_matrix((np.ones(len(row)), (row, col)), shape=(m, n), dtype=int)


#####################
# DRAWING FUNCTIONS #
#####################
def mol_with_canonical_index(mol, canon_indices):
    # canon_indices e.g. [3, 4, 2, 1,]
    n = mol.GetNumAtoms()
    for idx in range(n):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( int(canon_indices[idx]) ) )
    return mol

def mol_with_atom_index( mol ):
    n = mol.GetNumAtoms()
    for idx in range( n ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def mols_decorated(mols, option, canon_indices=None):
    decorateds = []
    for m in mols:
        x = copy.deepcopy(m)
        if 'canon' == option:
            decorateds.append(mol_with_canonical_index(x, canon_indices))
        elif 'atom' == option:
            decorateds.append(mol_with_atom_index(x))
        else:
            raise Exception('option {option} ???')
    return decorateds

######################
# MOLECULAR DISTANCE #
######################

def edge_multiset(mol, order={'C':0, 'H':1, 'O':2, 'P':3, 'S':4}):
    bonds = mol.GetBonds()
    bstrs = []
    for b in bonds:
        a1 = mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol()
        o1 = order[a1]
        a2 = mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()
        o2 = order[a2]
        bond_type = b.GetBondType()
        # order symbols:
        if order[a1] > order[a2]:
            bstr = a1+a2
        else:
            bstr = a2+a1
        bstr += str(bond_type)[:3] #3 chars
        # print(bstr)
        bstrs.append(bstr)
    return Multiset(bstrs)

def pospichal_kvanisnicka_distance(mol1, mol2):
    '''
    IOU on edge set. Assumes that mol1 and mol2 are connected components.
    Example:
        d = pospichal_kvanisnicka_distance(Chem.MolFromSmiles('CCOC(C)=P#COC=P#P'), Chem.MolFromSmiles('CCOC(C)P#COCP#P'))
    '''
    es1 = edge_multiset(mol1)
    es2 = edge_multiset(mol2)
    intersection = es1 & es2  #Counter({'OCSIN': 4, 'CCSIN': 2, 'PCTRI': 1, 'PPTRI': 1})
    # print(intersection); print(sum(intersection.values()))
    union = es1 | es2        #Counter({'OCSIN': 4, 'CCSIN': 2, 'PCDOU': 2, 'PCSIN': 2, 'PCTRI': 1, 'PPTRI': 1})
    # print(union); print(sum(union.values()))
    return 1.0* sum(intersection.values()) / sum(union.values())