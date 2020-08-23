1/ Only MPNN

- CUDA memory -> reduce batch, dimension
Cause I have too many nodes per molecule (atom node, edge node and msp node)
too many edges also, cause each msp node connect to all atom nodes

another way is not to use complete graph in MPNN stage

- Represent the node as size 3, where first element indicate the node type.