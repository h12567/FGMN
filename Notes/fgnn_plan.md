Yes. We can set each edge variable to take values 0,1,2,3. For each atom A, construct a factor that includes all edge 
variables connected to A.  The factor which is a function of the variables will take the value 1 if the sum of all the 
variables in it equals V(A) where V(A) is the valence of A and 0 otherwise. It should be possible to compute the messages
for loopy belief propagation for such factors efficiently using dynamic programming. For example, see LBP for parity 
checks (sum is even or odd depending on definition of parity used) in http://www.inference.org.uk/mackay/abstracts/mncEL.html, 
or more details in http://www.inference.org.uk/mackay/abstracts/mncN.html (I haven't worked through how they do it, but 
it should be by some form of dynamic programming).

We likely want some factors that exploit the mass spectrum as well, e.g. a factor for each element of the mass spectrum 
that is present when that element has large magnitude. The factor can include all the edge variables and be learned 
unlike the valence factors. It may also be useful (unclear) to introduce some latent vectors for the nodes to try to 
capture things we don't know enough to model within these factors.

Depending on the amount of training data available, may need to control the amount of parameter sharing in the model,
 like in Haroon's paper.
 
 
 
# Questions:

- If we formulate each molecule edge as a node, then find marginal distribution of the edge variable => OK

- Mass spectrum factor node?

- initialization? For parity-check problem, initially the bits are already chosen (so are the factor nodes or normal nodes).
How about this case?


# Self-beliefs:
https://www.cs.cmu.edu/~aarti/Class/10704_Spring15/lecs/lec25.pdf
- y_i stands for the mass spectrum nodes
- parity check equation takes other values besides 0


Extremely good repo:
https://github.com/krashkov/Belief-Propagation/