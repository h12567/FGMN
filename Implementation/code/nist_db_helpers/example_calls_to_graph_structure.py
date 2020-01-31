def get_observation(self):
        mol = copy.deepcopy(self.mol)
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        # n = mol.GetNumAtoms()
        # n_shift = len(self.possible_atom_types) # assume isolated nodes new nodes exist
        F = np.zeros((1, self.max_atom, self.d_n))
        valences = np.zeros((self.max_atom))
        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            atom_symbol = a.GetSymbol()
            implicit_valence = a.GetImplicitValence()
            if self.has_feature:
                formal_charge = a.GetFormalCharge()
                implicit_valence = a.GetImplicitValence()
                ring_atom = a.IsInRing()
                degree = a.GetDegree()
                hybridization = a.GetHybridization()
            # print(atom_symbol,formal_charge,implicit_valence,ring_atom,degree,hybridization)
            if self.has_feature:
                # float_array = np.concatenate([(atom_symbol ==
                #                                self.possible_atom_types),
                #                               (formal_charge ==
                #                                self.possible_formal_charge),
                #                               (implicit_valence ==
                #                                self.possible_implicit_valence),
                #                               (ring_atom ==
                #                                self.possible_ring_atom),
                #                               (degree == self.possible_degree),
                #                               (hybridization ==
                #                                self.possible_hybridization)]).astype(float)
                float_array = np.concatenate([(atom_symbol ==
                                               self.possible_atom_types),
                                              ([not a.IsInRing()]),
                                              ([a.IsInRingSize(3)]),
                                              ([a.IsInRingSize(4)]),
                                              ([a.IsInRingSize(5)]),
                                              ([a.IsInRingSize(6)]),
                                              ([a.IsInRing() and (not a.IsInRingSize(3))
                                               and (not a.IsInRingSize(4))
                                               and (not a.IsInRingSize(5))
                                               and (not a.IsInRingSize(6))]
                                               )]).astype(float)
            else:
                float_array = (atom_symbol == self.possible_atom_types).astype(float)
            # assert float_array.sum() == 6   # because there are 6 types of one
            # print(float_array,float_array.sum())
            # hot atom features
            # print(float_array, float_array.shape, atom_symbol)
            F[0, atom_idx, :] = float_array
            valences[atom_idx] = implicit_valence
        # add the atom features for the auxiliary atoms. We only include the
        # atom symbol features
        # auxiliary_atom_features = np.zeros((n_shift, self.d_n)) # for padding
        # temp = np.eye(n_shift)
        # auxiliary_atom_features[:temp.shape[0], :temp.shape[1]] = temp
        # F[0,n:n+n_shift,:] = auxiliary_atom_features

        # print('n',n,'n+n_shift',n+n_shift,auxiliary_atom_features.shape)

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))
        # for i in range(d_e):
        #     E[i,:n+n_shift,:n+n_shift] = np.eye(n+n_shift)
        for b in self.mol.GetBonds(): # self.mol, very important!! no aromatic
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)
            try:
                assert float_array.sum() != 0
            except:
                print('error',bond_type)
            E[:, begin_idx, end_idx] = float_array
            E[:, end_idx, begin_idx] = float_array
        ob = {}
        if self.is_normalize:
            E = self.normalize_adj(E)
        ob['adj'] = E                                             #shape   bond_types x max_atom x max_atom
        ob['node'] = F                                            #shape   max_atom x atom_types
        ob['spikes'] = np.array(self.mol_all_data['spikes'])      #shape   x_axis_len
        ob['valence'] = valences                                  #shape   max_atom

        distances, cc_labels, cc_nAtoms_labels, canon_labels, orbit_labels, nOrbit_labels = MassSpectrumEnv.graph_structure_properties(adj=np.copy(ob['adj']), mol=self.mol)
        ob['shortest_distances'] = distances                                                       #shape max_atom x max_atom
        zero_padding = np.zeros((self.max_atom - cc_labels.shape[-1]))                         #PLEASE NOTE ZERO PADDING!
        ob['connected_components'] = np.append(cc_labels, zero_padding, axis=-1)               #shape max_atom
        ob['connected_components_sizes'] = np.append(cc_nAtoms_labels, zero_padding, axis=-1)  #shape max_atom
        ob['canonical_indices'] = np.append(canon_labels + 1, zero_padding, axis=-1)           #shape max_atom  <NOTE +1>
        ob['orbit_indices'] = np.append(orbit_labels + 1, zero_padding, axis=-1)               #shape max_atom  <NOTE +1>
        ob['n_orbit_partitions'] = np.append(nOrbit_labels, zero_padding, axis=-1)             #shape max_atom
        return ob

    @staticmethod
    def graph_structure_properties(adj, mol):
        '''
        Compute a handful of properties of the graph's structure.
        INPUT:
            adj: Adjacency matrix 3 x n x n. The 3 come from 3 bond types. Each n x n slice contains 0's & 1's.
            mol: The rdkit molecule object.
        OUTPUT:
            see return
        '''
        # Compute APSP:
        A = np.sum(adj, axis=0) # n x n
        distances, _ = floydwarshall(A)
        mol.GetNumAtoms()

        # Compute orbits and canonical index for each connected component:
        A = np.copy(adj)
        A[1,:,:] = A[1,:,:]*2
        A[2,:,:] = A[2,:,:]*3
        A = np.sum(A, axis=0)  # n x n. Each elem has possibility 0,1,2,3
        cc_labels = connected_components(A, n_atoms=mol.GetNumAtoms())
        # print('cc_labels', cc_labels, len(cc_labels))
        cc_labels = np.array(cc_labels)
        cc_nAtoms_labels = np.ones_like(cc_labels) * -1 # number of atoms in each connected component
        canon_labels = np.ones_like(cc_labels) * -1 # canonical labelling
        orbit_labels = np.ones_like(cc_labels) * -1
        nOrbit_labels = np.ones_like(cc_labels) * -1
        n_cc = max(cc_labels) # n_cc number of connected components
        for i_cc in range(1, n_cc+1):
            '''
            Find indices of atoms in this connected component:
            '''
            vs = np.where(cc_labels==i_cc)[0]
            # print('index of CC:', i_cc, 'has v\'s', vs)
            nAtoms = len(vs) 
            cc_nAtoms_labels[vs] = nAtoms
            # if 1 == nAtoms:
            #     continue #skip single stand-alone atom
            '''
            Subset rows & colums of adjacency matrix corresponding to atoms in this connected component to get another adjacency matrix:
            '''
            A_cc = A[list(vs), :]
            A_cc = A_cc[:, list(vs)]
            # print('A_cc', A_cc)
            cc_node_list = [mol.GetAtomWithIdx(int(v)).GetSymbol() for v in vs]  #['C', O', 'C', 'C', 'C']
            # print('cc_node_list', cc_node_list)
            '''
            Create molecule of this connected component::
            '''
            mol_cc, _ = mol_from_graph(cc_node_list, list(A_cc))
            mol_cc.UpdatePropertyCache()
            # Compute canonical indexing:
            canon_order = canonicalize(mol_cc)
            # print_canonical_order(mol_cc)
            # make labels:
            for v, c in zip(vs, canon_order):
                canon_labels[v] = c
            
            # Compute orbits:
            orbits, n_orbits = compute_orbits(cc_node_list, A_cc)
            # make labels:
            for v, o in zip(vs, orbits):
                orbit_labels[v] = o
                nOrbit_labels[v] = n_orbits
        '''
        distances,        # n x n shortest distances (# hops) between atoms.
        cc_labels,        # [1 1 1 1 2 3 4 4 4 4 4 2 5 2]
        cc_nAtoms_labels, # [4 4 4 4 3 1 5 5 5 5 5 3 1 3]
        canon_labels,     # [0 3 2 1 0 0 0 3 4 1 2 2 0 1]
        orbit_labels,     # [0 1 2 3 0 0 0 1 2 3 4 1 0 0]
        nOrbit_labels     # [4 4 4 4 2 1 5 5 5 5 5 2 1 2]
        '''
        return distances, cc_labels, cc_nAtoms_labels, canon_labels, orbit_labels, nOrbit_labels