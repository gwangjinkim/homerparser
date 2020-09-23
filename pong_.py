# pong from pixels
# http://karpathy.github.io/2016/05/31/rl/
# very long good explanations!

# 210x160x3 byte array  100, 800 numbers total
# move decide: UP or DOWN
# ball went past opponent +1 reward
# we missed ball          -1 reward
# otherwise                0 reward

# compute hidden layer neuron activations
h = np.dot(W1, x)
h[h<0] = 0 # ReLU nonlinearity: thershold at zero
logp = np.dot(W2, h) # compute log prob of going up
p = 1.0 / (1.0 + np.exp(-logp)) # sigmoid function (gives probability of going up)

# W1 and W2 are two matrices we initialize randomly

# at least 2 frames to policy network -> motion
# preprocessing -> feed DIFFERENCE FRAMES to the network (subtraction current - last frame)

code
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5





















#############################
# pong keras
#############################
# https://medium.com/gradientcrescent/fundamentals-of-reinforcement-learning-automating-pong-in-using-a-policy-model-an-implementation-b71f64c158ff

"""
!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!apt update > /dev/null 2>&1
!apt install --upgrade setuptools 2>&1
!pip install ez_setup > /dev/null 2>&1
!pip install gym[atari] > /dev/null 2>&1

"""

import numpy as np
import gym

# gym initialization
env = gym.make("Pong-v0")
observation = env.reset()
prev_input = None

# declaring two actions that happen in pong ofr an agent, UP or DOWN
# declaring 0 means stay still. Note this is pre-defined specific to package
UP_ACTION = 2
DOWN_ACTION = 3

# hyperparameter - gamma - measure of effect of future events
gamma = 0.99

# initialize variable used in main loop
x_train, y_train, rewards = [], [], []
reward_sum = 0
episode_nb = 0


# take look at game in action
import matplotlib.pyplot as plt

env = gym.make("Pong-v0") # env info
observation = env.reset()
# ball is released after 20 frames
for i in range(22):
    if i > 20:
        plt.imshow(observation)
        plt.show()
observation, _, _, _ = env.step(1)

# cutting away unnecessary areas

def prepro(I):
    """
    prepro 210x160x3 frame into 6400 (80x80) 1D float vector
    """
    I = I[35:195] # crop
    I = I[::2, ::2, 0] # downsample by factor 2
    I[I==144] = 0 # erase background (type 1)
    I[I==109] = 0 # erase background (type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

# show preprocessed
obs_preprocessed = prepro(observation).rshape(80, 80)
plt.imshow(obs_preprocessed, cmap="gray")
plt.show()

def discount_rewards(r, gamma):
    """Take 1D float array of rewards and compute discounted reward"""
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0 # if game ended (in pong), reset
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        discounted_r -= np.mean(discounted_r) # normalizing result
        discounted_r /= np.std(discounted_r) # idem using standard dev
        return discounted_r

# reward (or punishment) is spread across a number of frames to judge the appropriateness of our actions)


https://medium.com/@adrianitsaxu






















