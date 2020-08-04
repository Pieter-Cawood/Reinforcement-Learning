# DP
### Modify the gridworld you created last week to create a 4x4 version <br>

• No obstacles <br>
• Rewards of -1 on all transitions <br>
• Goal in the top left corner – entering the goal state ends the episode <br>

### 1. First we consider how to compute the state-value function <img src="https://render.githubusercontent.com/render/math?math=v_\pi"> for an arbitrary policy <img src="https://render.githubusercontent.com/render/math?math=\pi">

Given this environment and a uniform random policy, implement 2 versions of policy evaluation:<br>
(For both versions: The initial approximation, <img src="https://render.githubusercontent.com/render/math?math=v_0">, is chosen
arbitrarily (except that the terminal state, if any, must be given value 0)<br>
**Version 1** <br>
A two-array version, which only updates the value function after looping through all states 

![Figure 1-2](iterative-policies.PNG "Figure 1-2")
![Figure 1-3](2array.png "Figure 1-3")

**Version 2** <br>
• An in-place version with a threshold value of 𝜃 = 0.01 <br>

![Figure 1-1](in-place.png "Figure 1-1")

• NB: Careful of how you handle the terminal state!! <br> 
• For a given 𝛾, record the number of iterations of policy evaluation until convergence <br>

**Submit**
1. A 2d heatmap plot of the value function for 𝛾 = 1
2. A combined plot of both versions of policy evaluation for different discount rates
1. The 𝑥-axis should be the discount rate. The range of discounts should be specified by np.logspace(- 0.2, 0, num=20)
2. The 𝑦-axis should be the number of iterations to convergence
