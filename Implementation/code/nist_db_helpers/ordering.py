# This class is responsible for atoms ordering functions to remove symmetry problem.
import numpy as np

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

# E is always (13, 13), so we want to extract sub-matrix of it before calculating the orders.
class OrderingBase:
    @classmethod
    def _extract_sub_adj_matrix(cls, E, num_atoms):
        return E[:num_atoms, :num_atoms]

    @classmethod
    def _swap_row(cls, E, i, j):
        temp = E[i, :]
        E[i, :] = E[j, :]
        E[j, :] = temp
        return E

    @classmethod
    def _swap_col(cls, E, i, j):
        temp = E[:, i]
        E[:, i] = E[:, j]
        E[:, j] = temp
        return E

class SvdOrdering(OrderingBase):
    @classmethod
    def order(cls, E, vertex_arr):
        # return list of idxes order and updated_adj_matrix
        sub_E = cls._extract_sub_adj_matrix(E, len(vertex_arr))
        u, s, v = np.linalg.svd(sub_E)
        atoms_repr = np.dot(u, np.diag(s))
        # iterate through each atom types
        current_adj_row_idx = 0
        for atom_type in range(0, max(vertex_arr)+1):
            row_idxes = np.where(np.array(vertex_arr) == atom_type)[0]

            #get orders of the atoms_reprs vectors
            # TODO: Experiment different sorting criterias???
            sort_idxes = argsort(atoms_repr[row_idxes].tolist())

            # doing selection sort
            for i in range(len(row_idxes)):
                # Because we need to sort the vertex_arr in order of atom_type, so group C together
                # group O together for example. Therefore, need to re-arrange adj_matrix E as well.
                E = cls._swap_row(E, current_adj_row_idx, sort_idxes[i])
                E = cls._swap_col(E, current_adj_row_idx, sort_idxes[i])
                current_adj_row_idx += 1

        vertex_arr.sort()
        return E, vertex_arr
