# Cliffwalking

Use the CliffWalking domain from OpenAI gym
• See Example 6.6, pg 132 in Sutton and Barto [2018]
• Modify the TD(𝜆) algorithm presented to implement SARSA(𝜆)
• The only difference here is that there is an eligibility trace for each state-action
pair!
• See the first edition of Sutton and Barto for more info
• Use 𝜀-greedy policies with 𝜀 = 0.1 and a learning rate of 𝛼 = 0.5
• Run SARSA(𝜆) on the domain for 𝜆 = {0, 0.3, 0.5, 0.7, 0.9} for 500 episodes
• Record the current estimate of the Q-value function after each episode

