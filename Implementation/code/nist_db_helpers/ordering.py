# This class is responsible for atoms ordering functions to remove symmetry problem.
import numpy as np
from numpy.linalg import eig as eigenValuesAndVectors

def argsort(seq, eigenvectors_sort_indices):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=lambda i: np.array(seq.__getitem__(i))[eigenvectors_sort_indices].tolist())

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

    @classmethod
    def _swap_vertex(cls, vertex_arr, i, j):
        temp = vertex_arr[i]
        vertex_arr[i] = vertex_arr[j]
        vertex_arr[j] = temp
        return vertex_arr

class SvdOrdering(OrderingBase):
    @classmethod
    def order(cls, E, vertex_arr):
        # return list of idxes order and updated_adj_matrix
        sub_E = cls._extract_sub_adj_matrix(E, len(vertex_arr))
        u, s, v = np.linalg.svd(sub_E)
        atoms_repr = np.dot(u, np.diag(s))
        # iterate through each atom types
        current_adj_row_idx = 0

        '''
        steps
        A^T A get eigenvalues, order is order of eigenvectors in V
        use that to sort the sort_idxes
        '''
        mat = np.dot(np.transpose(sub_E), sub_E)
        solution = eigenValuesAndVectors(mat)
        eigenvalues = solution[0]
        eigenvectors_sort_indices = np.argsort(eigenvalues)

        for atom_type in range(0, max(vertex_arr)+1):
            row_idxes = np.where(np.array(vertex_arr) == atom_type)[0]

            #get orders of the atoms_reprs vectors
            # TODO: Experiment different sorting criterias???
            sort_idxes = argsort(atoms_repr[row_idxes].tolist(), eigenvectors_sort_indices)

            # doing selection sort
            for i in range(len(row_idxes)):
                # Because we need to sort the vertex_arr in order of atom_type, so group C together
                # group O together for example. Therefore, need to re-arrange adj_matrix E as well.
                E = cls._swap_row(E, current_adj_row_idx, row_idxes[sort_idxes[i]])
                E = cls._swap_col(E, current_adj_row_idx, row_idxes[sort_idxes[i]])
                vertex_arr = cls._swap_vertex(vertex_arr, current_adj_row_idx, row_idxes[sort_idxes[i]])
                current_adj_row_idx += 1

        vertex_arr.sort()
        return E, vertex_arr
