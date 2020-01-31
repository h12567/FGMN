This repo contains crawled NIST data in `nist_database` directory.

The 3 .py files contain code excerpts for loading the files (molecule .mol and mass spectrum .jdx). As these are excerpts copied and pasted from another codebase, effort is needed in glue-ing them together. But should be just a little bit of effort.

`util.py` contains relatively low level loading functions among other functions.

`loader.py` is a high level loader (can accomodate filter parameters e.g. load molecules less than n atoms)

`get_nist_data_subset.py` shows example of how the function in `loader.py` is called.

`graph_structure.py` contains functions to compute graph properties (like shortest distances, canonical indices, etc.)

`example_calls_to_graph_structure.py` contains examples of calling the functions in `graph_structure.py`. As these are code snippets copied and pasted from another codebase, some effort is required to make them runnable. But they should give a general idea of how to call the functions in `graph_structure.py`.