# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/


# actionspace

# 1. south
# 2. north
# 3. east
# 4. west
# 5. pickup
# 6. dropoff

# Taxi-V2


#############################
# install
#############################

!pip install cmake 'gym[atari]' scipy

import gym

env = gym.make("Taxi-v3").env
# using .env to avoid training stopping at 200 iterations 
# - default in new Gym

env.render()

# reset env

env.reset()

# step env by one timestep

env.step(action)

# returns:
# observation
# reward
# done (successuffly picked up and dropped off a passgenger - episode
# info (performance, latency for debugging)

# render one fram of env (visualization)
env.render()

"""
there are 4 locations (labeled with different letters).
our job is to pick up passenger at one location and drop him off at another
receive +20 points for successful drop-off
every time step it takes -1 point
illegal pick-up and drop-off actions -10 points
"""


# reset to a new random state
env.reset()
env.render()

print(f"Action Space {env.action_space}")
print(f"State Space {env.observation_space}")

# - yello square -> taxi
# - "|"          -> wall
# - R, G, Y, B   -> possible pickup and destination locations
#                -> current pick up: Blue letter
#                -> current destination: purple

# action space size 6
# state space size  500

# 0 -> south
# 1 -> north
# 2 -> east
# 3 -> west
# 4 -> pickup
# 5 -> dropoff

# taxi at row 3 col 1
# passenger at location 2
# destionation is location 0

state = env.encode(3, 1, 2, 0) # (taxi row, taxi col, passenger index, destination index)
state # 328 -> a number 0 <= x <= 499

list(env.decode(328)) # [3, 1, 2, 0]

# set by state number
# doesn't work: env.env.s

env.desc # gives numpy array of ascii pic


# initial reward table called 'P'
# a matrix that has number of states as rows
# number of actions as columns
# states x actions matrix

# since every state is in this matrix, we can see the default reward values assigned to our
# illustration's state:
env.P[328]
# {0: [(1.0, 428, -1, False)],
#  1: [(1.0, 228, -1, False)],
#  2: [(1.0, 348, -1, False)],
#  3: [(1.0, 328, -1, False)],
#  4: [(1.0, 328, -10, False)],
#  5: [(1.0, 328, -10, False)]}
# 
# {action: [(probability, nextstate, rward, done)]}

# in this env probability is always 1.0
# nextstate is state we would be in if we take the action at this index of the dict
# all the movement actions have a -1 reward and the pickup/dropoff actions have -10 reward
# in this particular state
# if in a state where taxi has a passenger and is on top of the right destination, we would see
# reward of 20 at dropoff action (5)

# 'done' is used to tell us when we have successfully dropped off a passenger
# in the right location
# each successfull dropoff is the end of an episode


####################################
# solution without RL
####################################

# let program use P table only


env.s = 328 # set env to illustration's state

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    
    if reward == -10:
        penalties += 1

    # put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    })
    
    epochs += 1

print(f"Timesteps taken: {epochs}")
print(f"Penalties incurred: {penalties}")

# Timesteps taken: 5201
# Penalties incurred: 1715


# visualize

from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

print_frames(frames)



##################################
# enter RL
##################################

# Q-table stores Q-values which map a (state, action) combination
# Q-value for a particular state-action combination is "quality" of an action
# better Q-value -> greater rewards

# e.g. when passenter at current destination, Q-value for 'pickup' is higher

# initial Q-values are arbitrary
# but gets updated:

# Q(state, action) <- (1-alpha)*Q(state,action) + alpha*(reward + gamma*maxQ_a(next_state, all_actions))

# alpha is learning rate (0 < alpha <= 1)
# gamma is discount factor (0 <= gamma <= 1)
#    how much importance to give to future rewards
#    ~ 1 -> long-term effective award
#    ~ 0 -> only immediate award (greedy)


# Q-table is a matrix 
# row states (500) x col actions (6)
# initial values: 0

# values are updated after training

# optimize the agent's traversal through the environment for maximum rewards

"""
0. initialize Q-table by all zeros
1. start exploring actions: For each state, select any one among all possible actions
   for the current state (S).
2. travel to the next state (S') as a result of that action (a).
3. for all possible actions from the state (S') select the one with the highest Q value.
4. update Q-table values using the equation.
5. set the next state as the current state.
6. if goal state is reached, then end and repeat the process.

after enough random exploration of actions,
Q-table will converge to a action-value function which can exploit to pick the most
optimal action from a given state.

There's a tradeoff between exploration (choosing random action)
and exploitation (choosing actions based on already learned Q-values).

We want to prevent action from always taking the same route,
and possibly overfifting.

So epsilon is introduced to cater to this during training.

Instead of just selecting the best learned Q-value action, we sometimes favor
exploring the action space further.

Lower epsilon value results in episodes with more penalties (on average)
- obvious because we are exploring and making random decisions.
"""


#########################################
# Implementing Q-learning in python
#########################################

# training the agent

import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])



%time
"""Training the agent"""

import random
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# for plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()
    
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # explore action space
        else:
            action = np.argmax(q_table[state]) # exploit learned values
        
        next_state, reward, done, info = env.step(action)
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1
    
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
        
print("Training finished.\n")

# Now that Q-table has been established over 100000 episodes
q_table[328]
# array([ -2.40230653,  -2.27325184,  -2.40828582,  -2.3599973 ,
#        -10.81195156,  -9.42292528])


# so north is best













