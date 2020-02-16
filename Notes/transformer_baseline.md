##  Encoder + EdgeClassifier method
K: max_atoms in a molecule
M: length of a mass spectrum
T: number of bonds types (no bond, single, double, triple)
vocabulary size of the encoder embedding input is that largest value in mass spectrum

- nist_db_helpers/prepare_train_dataset.py: Main file for training data preparation
    + filter out by ester group, at most 4 Carbons in R part, and only contains ("C", "H", "O", "N", "P", "S")
    + vertex_arr.npy: list of atoms indices. e.g (0, 2, 1, 0, 1, 2)
    + mol_adj_arr.npy: adjacency matrix of atoms edges (note that an entry in matrix will be a vector 4 x 1 (no bond, single bond, double bond, triple bond))
    + msp_arr.npy: Mass spectrum
    
- transformers/train.py: Entry file for training
    + Encoder input will be a concatenation of vertex_arr and msp_arr from above. 
        Input Shape: (batch_size, K + M)
        Output Shape: (batch_size, K,  encoder_output_size)
    + Encoder output will be input to edge classifier, which will do some matrix manipulation to
        group embeddings of two nodes into a single entry. 
          Input Shape: (batch_size, K, encoder_output_size)
          Output Shape: (batch_size, K, K, T)
    
## NOTE: 
- Currently training not converging, need to tune or find another way.

- Try changing the order of vertices in vertex_arr.npy and mol_adj_arr.npy. **QUESTIONS**: Prof Lee
suggested that we can use canonical form as a sequential order. But to find the canonical representation of a graph,
we need to know the graph structure, which cannot be done in testing stage. Is this reasoning correct?

