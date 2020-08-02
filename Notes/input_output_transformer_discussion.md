Rethink definition of vocabulary size.

Let's say atom is [0, 0, 1, 0,    1, 0 ,0 , 0]
13 atoms 

transform 800 to (13, k) or (k ,8)? feasible?  


# INPUT PROBLEM:

## Let's say (13, k + 8) (combine msp and atoms)
then should it be 800 -> (1,k) and duplicate 13 times. Or 800 -> (13,k) ?

First approach cons: k is much bigger than 8, so all inputs are like almost the same.

Second approach cons: needs to find an order of assigning each of 13 msp vectors to each of 13 atoms (maybe in canonical order)? 
Note: Doesn't matter if each of the 13 msp component does not contain all information, because later in self-attention they will attend each other.


## Let's say (13+k, 8) 
-> Cons: mix of 13 entries of types atoms and k entries of types mass spectrum mismatch so maybe a problem.


# OUTPUT PROBLEM: 

## Use SVD to sort the atoms in ground truth
You’d want to minimise the symmetry — if one atom is indistinguishable from another atom, then the embedding would also be the same, and the edge classifier will also have to treat the two atoms the same and would likely be unable to fit even the training data.
So the first carbon should be distinguished from the second carbon, etc. — call that the position of the atom of that type. Each atom should be identified as a pair — atom type and the its position. Compute a different embedding for each pair.
Unfortunately, the atoms of the same type in the labeled graph are not labeled with position to match our labelling of position in the input. One way to handle that is to do reinforcement learning and sequentially output the atoms with the policy deciding the ordering. But may be simpler to try labelling them with a fixed labelling function for imitation learning. One possibility is to use SVD on the adjacency matrix. The output of SVD is an embedding vector for each atom. Can sort the atoms of each type lexicographically (by coefficient of first eigenvector first, then if tie, by coefficient of second eigenvector, ….) to determine its ordering.


Use the rows of U.S to represent the nodes. To compute the entry (i,j) of A, you can do an inner product of row i of U.S with column j of V^T. So row i of U.S contains all the information required to reproduce row i of A, given V^T.
What we would like is to have common substructures in different molecules to be labeled the same way so that it generalises more easily — hopefully something like this would approximate that.


## In last layer, if we just use fully connected layer then cannot control the weight => cannot guarantee symmetry => need change.

We discussed putting in the known ester subgraph yesterday. We can try to encode that as additional inputs, or possibly as done in https://www.kaggle.com/c/champs-scalar-coupling/discussion/106575

Predicting Molecular Properties | Kaggle
Great write up, and impressive that you built a transformer with priors extracted from domain knowledge. despite the fact that I will never, ever, forgive the original paper for the monstrosity that is the "query", "key", and "value" terminology for self-attention layers ;-)
www.kaggle.com
we can try to encode it in the attention mechanism. They added another term into the attention mechanism – we can also try something similar, e.g. have a learnable weight vector that we use to inner product with one-hot vector of single, double, triple bond, no-bond (with no addition term if nothing is known). This may also be useful when sequentially labeling the output using reinforcement learning as we can add the bonds for the partially labeled graph. For reinforcement learning, the reward function will need some thought.

They also used a variable for each edge, instead of each node. That is also something to consider.

https://papers.nips.cc/paper/9367-graph-transformer-networks.pdf
https://www.kaggle.com/c/champs-scalar-coupling/discussion/106575
https://arxiv.org/pdf/2002.08264.pdf (Molecule attention network)
https://github.com/boschresearch/BCAI_kaggle_CHAMPS/blob/master/models/model_A/graph_transformer.py
https://arxiv.org/pdf/1602.06289.pdf

