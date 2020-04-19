# This class is responsible for atoms ordering functions to remove symmetry problem.
import numpy as np
from numpy.linalg import eig as eigenValuesAndVectors
import copy

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
        temp = copy.deepcopy(E[i, :])
        E[i, :] = copy.deepcopy(E[j, :])
        E[j, :] = copy.deepcopy(temp)
        return E

    @classmethod
    def _swap_col(cls, E, i, j):
        temp = copy.deepcopy(E[:, i])
        E[:, i] = copy.deepcopy(E[:, j])
        E[:, j] = copy.deepcopy(temp)
        return E

    @classmethod
    def _swap_elem_in_row(cls, E, row_i, col_1, col_2):
        temp = E[row_i, col_1]
        E[row_i, col_1] = E[row_i, col_2]
        E[row_i, col_2] = temp
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
        old_E = copy.deepcopy(E)
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
        assert np.array_equal(E, old_E)
        for atom_type in range(0, max(vertex_arr)+1):
            row_idxes = np.where(np.array(vertex_arr) == atom_type)[0]

            assert len(list(filter(lambda x: x < current_adj_row_idx, row_idxes))) == 0

            #get orders of the atoms_reprs vectors
            # TODO: Experiment different sorting criterias???
            sort_idxes = argsort(atoms_repr[row_idxes].tolist(), eigenvectors_sort_indices)

            # doing selection sort
            for i in range(len(row_idxes)):
                # Because we need to sort the vertex_arr in order of atom_type, so group C together
                # group O together for example. Therefore, need to re-arrange adj_matrix E as well.
                # E = cls._swap_row(E, current_adj_row_idx, row_idxes[sort_idxes[i]])
                to_swap_idx = row_idxes[sort_idxes[i]]
                assert to_swap_idx >= current_adj_row_idx
                # new_E[current_adj_row_idx, :] = E[row_idxes[sort_idxes[i]], :]
                # E = cls._swap_col(E, current_adj_row_idx, row_idxes[sort_idxes[i]])
                E = cls._swap_row(E, current_adj_row_idx, to_swap_idx)
                # new_E[:, current_adj_row_idx] = E[:, row_idxes[sort_idxes[i]]]
                # E = cls._swap_elem_in_row(E, current_adj_row_idx, current_adj_row_idx, to_swap_idx)
                # E = cls._swap_elem_in_row(E, to_swap_idx, current_adj_row_idx, to_swap_idx)
                E = cls._swap_col(E, current_adj_row_idx, to_swap_idx)

                vertex_arr = cls._swap_vertex(vertex_arr, current_adj_row_idx, to_swap_idx)
                row_idxes = list(map(lambda x: to_swap_idx if x == current_adj_row_idx else x, row_idxes))
                current_adj_row_idx += 1

        # Test that diagonals all zero
        for i in range(len(E)):
            assert E[i, i] == 0


        rows_sum_E, cols_sum_E, rows_sum_old_E, cols_sum_old_E = [], [], [], []
        # Test that list of row sums are the same in ascending order
        for i in range(len(E)):
            rows_sum_E.append(np.sum(E[i, :]))
            rows_sum_old_E.append(np.sum(old_E[i, :]))
            rows_sum_old_E.sort()
            rows_sum_E.sort()
        assert np.array_equal(rows_sum_old_E, rows_sum_E)

        # Test that list of column sums are the same in ascending order
        for j in range(len(E)):
            cols_sum_E.append(np.sum(E[:, j]))
            cols_sum_old_E.append(np.sum(old_E[:, j]))
            cols_sum_E.sort()
            cols_sum_old_E.sort()
        assert np.array_equal(cols_sum_old_E, cols_sum_E)

        # Test that vertex_arr is in ascending order
        dummy_vertex_arr = copy.deepcopy(vertex_arr)
        dummy_vertex_arr.sort()
        assert np.array_equal(vertex_arr, dummy_vertex_arr)

        return E, vertex_arr
