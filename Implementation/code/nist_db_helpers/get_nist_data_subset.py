import os
import sys
import time
import datetime

if __name__ == "__main__":
    from loader import get_useable_nist_data
    fnames, len_mass_spectrum_x_axis, more_info = get_useable_nist_data(force_recompute=True, 
                                                                        possible_atoms=('C','H','O','P','S'), 
                                                                        min_n_atoms=3,
                                                                        limit_n_atoms=8,
                                                                        whole_molecule_0_implicit_valence = True,
                                                                        path_save_info='nist_data_subset_CHOPS_nAtoms3-8.txt'   # info is saved here.
                                                                        )
    print(more_info)
