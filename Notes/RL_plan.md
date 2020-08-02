Policy_Gradient
Encoder: Input--Mass Spectrum k [m/z, relative redundancy] and do embedding（180）k  180
Decoder: Input--Edges 78*[edge type 1 2 3([0,0,0] means emtpy, [1,0,0] means 1 bond...), atom1-mass, atom2-mass, possible-edge type C, O [1,1,0]]. Initially, edge type are all [0,0,0] except that the first two C=O, C--O. [1 1 0 … 0] 13

When training, every time we output 78*4(+stop action) output after softmax and select the action with the highest probability as the next the edge. Update the graph and use it as the new input to the decoder. Repeat the decoder and update the graph until we sample enough edges number. 
For example, we sample N times. Now we have N [actions] and N [rewards]. If the new action exists in the target graph, the returned reward is 1, otherwise, the reward is 0. We take log of actions and get N [log_probs]. For the loss function, negative_log_policy_gradient= -sum( [log_actions] * [rewards]). Doing permutation and find one with the lowest loss and do backward then.
I also calculate the accuracy of each edge matrix. This time, I am only concerned about the labels in the target matrix that are non-zero. If there are 10 non-zero 0, and 4 of them are the same in both target and predict matrix, then the acc is 0.4. Since in ester, C=O, C-O are determined, so acc is always larger than 0.
  

{
    "trip_id": {
        [
            {
                "rawlat": 0.1234,
                "rawlng": 12.345,
                "timestamp": 1,
                "accuracy_level": 3,
                "speed": 1000,
                "street_id": 12
            },
            {
                "rawlat": 1.1234,
                "rawlng": 13.345,
                "timestamp": 2,
                "accuracy_level": 3,
                "speed": 809,
                "street_id": 13
            }
        ]
    }
}


Questions:
"I am only concerned about the labels in the target matrix that are non-zero": Rationale??? 
[0.22, 0.38, 0.33, 0.33, 0.33, 0.29, 0.29, 0.29]

# Policy Gradient Summary
towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c

towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d      v     

# Policy Gradient Simple
Gradient ascent only on the most optimal choice at each step:

$\theta_{t+1} = \theta_t + \alpha \nabla \pi_{\theta_t}(a^* | s)$

# Weighing the gradient
$\theta_{t+1} = \theta_t + \alpha \hat{Q}(s, a) \nabla \pi_{\theta_t}(a | s)$

# Sample based on frequency of action probability
previously choose an action to update by random

now choose by probability of action under policy

$\theta_{t+1} = \theta_t + \alpha \frac{\hat{Q}(s, a) \nabla \pi_{\theta_t}(a | s)}{\pi_\theta(a|s)}$


FINAL:
Equal as
$\theta_{t+1} = \theta_t + \alpha \hat{A}(s, a) \nabla_\theta log(\pi_\theta(s|a))$
