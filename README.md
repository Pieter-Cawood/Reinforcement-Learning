# Cliffwalking

http://sabinemaennel.ch/udacity-deeplearning/Temporal_Difference.html

## Use the CliffWalking domain from OpenAI gym <br>
• See Example 6.6, pg 132 in Sutton and Barto [2018]<br>
## Modify the TD(𝜆) algorithm presented to implement SARSA(𝜆)
• The only difference here is that there is an eligibility trace for each state-action
pair!<br>
• See the first edition of Sutton and Barto for more info<br>
• Use 𝜀-greedy policies with 𝜀 = 0.1 and a learning rate of 𝛼 = 0.5<br>
• Run SARSA(𝜆) on the domain for 𝜆 = {0, 0.3, 0.5, 0.7, 0.9} for 500 episodes<br>
• Record the current estimate of the Q-value function after each episode<br>


Perform a single run of the algorithm. After each episode plot the
value function (take max
𝑎
𝑄(𝑠, 𝑎)) learned so far as a heatmap for
each 𝜆 side by side. Ensure the visualisation aligns with the layout of
the domain. This should result in 500 separate plots/images. Turn
these images into an animation/video and submit it.
