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

















