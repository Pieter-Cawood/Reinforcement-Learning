Training the agent:
Run actor-critic.py to train an agent.
This will generate some graphics plus two .pkl files,one 
for the neural network's parameters (i.e. initialisation),
one for the neural network's trained state.
NOTE! The first three lines of code are to install matplotlib
in the Docker environment (since it didn't come pre-installed).
You can remove these if you don't need them.

Running the trained agent:
Run test.py to run a trained agent for an episode. The script
will print out the final score, a list of the positive rewards,
and how many actions it took during the episode.