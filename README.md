# DP
 **Modify the gridworld you created last week to create a 4x4 version** <br>

• No obstacles <br>
• Rewards of -1 on all transitions <br>
• Goal in the top left corner – entering the goal state ends the episode <br>
• Given this environment and a uniform random policy, implement 2 versions of policy evaluation <br>
• The in-place version, as presented in the book <br>
• A two-array version, which only updates the value function after looping through all states (see pg 75) <br>
• Use a threshold value of 𝜃𝜃 = 0.01 <br>
• NB: Careful of how you handle the terminal state!! <br>
• For a given 𝛾𝛾, record the number of iterations of policy evaluation until convergence <br>

**Submit**
1. A 2d heatmap plot of the value function for 𝛾𝛾 = 1
2. A combined plot of both versions of policy evaluation for different discount rates
1. The 𝑥𝑥-axis should be the discount rate. The range of discounts should be specified by np.logspace(- 0.2, 0, num=20)
2. The 𝑦𝑦-axis should be the number of iterations to convergence
