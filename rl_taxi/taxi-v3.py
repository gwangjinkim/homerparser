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






















