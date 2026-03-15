# Coding4Medicine
AI-Optimized Path Planning for 3D Bioprinted Tumor Models

This simulation demonstrates a hybrid machine learning approach to bioprinter path planning, combining systematic grid traversal with Q-learning — a model-free reinforcement learning algorithm.  It is a method from Reinforcement Learning, which is a branch of machine learning focused on training agents to make sequences of decisions by interacting with an environment. The agent learns through trial and error, receiving rewards or penalties for actions, and aims to maximize cumulative reward over time.

The bioprinter nozzle sweeps every cell in a raster pattern (row by row), ensuring full grid coverage each episode. At each cell, a Q-learning agent decides whether to print or skip based on a learned value function. The Q-table maps every (position, cell-type) state to expected future rewards for each action, updated each step using the Bellman equation:
<bold>Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s',a') − Q(s,a) ]</bold>

where α is the learning rate, γ is the discount factor, r is the immediate reward, and s' is the next state.
Early episodes use ε-greedy exploration (ε = 1.0), meaning the agent acts randomly to discover which cells should be printed. Epsilon decays each episode, gradually shifting the agent from exploration toward exploitation of its learned Q-values. By episode 8–10, the agent reliably prints only on tumor target cells, achieving 16/16 accuracy.


### The Al maps every (state, action) pair to a value called Q.
- High Q → the AI favors that move.
- Negative Q → learned to avoid it.

## How to run
1. Install VSCode
2. Open terminal, git clone this repository into the desired directory, and set it up by typing the following:
- <code>mkdir NAME_OF_DIRECTORY</code>
- <code> cd NAME_OF_DIRECTORY</code>
- <code> git clone https://github.com/avikaprasad22/Coding4Medicine.git </code>
- <code>cd Coding4Medicine </code>
- <code>python3 -m venv venv</code>
- <code>source venv/bin/activate</code>
- <code>pip install numpy matplotlib</code>
- <code>code .</code>

### Now that you have the project open in VSCode:
1. Go to the terminal and type <code>python bioprint_simulator.py</code>
2. Enjoy the simulation!
3. Press Ctrl C on the keyboard to stop the simulation or just X out.

