'''
- Temporal Difference Learning and Control Tutorial 
- Reinforcement Learning Tutorial
- On-Policy SARSA (State-Action-Reward-State-Action) Reinforcement Learning 

Author: Aleksandar Haber
Date: January 2023

- This Python file contains driver code for the SARSA Temporal Difference Learning Algorithm
- This Python file imports the class SARSA_Learning that implements the algorithm 
- The definition of the class SARSA_Learning is in the file "functions.py"


'''
# Note: 
# You can either use gym (not maintained anymore) or gymnasium (maintained version of gym)    
    
# tested on     
# gym==0.26.2
# gym-notices==0.0.8

#gymnasium==0.27.0
#gymnasium-notices==0.0.1

# classical gym 
import gym
# instead of gym, import gymnasium 
# import gymnasium as gym
import numpy as np
import time
from functions import SARSA_Learning
 
# create the environment 
# is_slippery=False, this is a completely deterministic environment, 
# uncomment this if you want to render the environment during the solution process
# however, this will slow down the solution process
#env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode="human")

# here we do not render the environment for speed purposes
env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env.reset()
# render the environment
# uncomment this if you want to render the environment
#env.render()
# this is used to close the rendered environment
#env.close()
 
# investigate the environment
# observation space - states 
env.observation_space
 
env.action_space
# actions:
#0: LEFT
#1: DOWN
#2: RIGHT
#3: UP


# define the parameters

# step size
alpha=0.1
# discount rate
gamma=0.9
# epsilon-greedy parameter
epsilon=0.2
# number of simulation episodes
numberEpisodes=10000

# initialize
SARSA1= SARSA_Learning(env,alpha,gamma,epsilon,numberEpisodes)
# simulate
SARSA1.simulateEpisodes()
# compute the final policy
SARSA1.computeFinalPolicy()

# extract the final policy
finalLearnedPolicy=SARSA1.learnedPolicy

# simulate the learned policy for verification
while True:
    # to interpret the final learned policy you need this information
    # actions: 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
    # let us simulate the learned policy
    # this will reset the environment and return the agent to the initial state
    env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode='human')
    (currentState,prob)=env.reset()
    env.render()
    time.sleep(2)
    # since the initial state is not a terminal state, set this flag to false
    terminalState=False
    for i in range(100):
        # here we step and return the state, reward, and boolean denoting if the state is a terminal state
        if not terminalState:
            (currentState, currentReward, terminalState,_,_) = env.step(int(finalLearnedPolicy[currentState]))
            time.sleep(1)
        else:
            break
    time.sleep(0.5)
env.close()

