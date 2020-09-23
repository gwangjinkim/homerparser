# https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning
# https://www.kaggle.com/alexisbcook/play-the-game

"""
!pip install kaggle
!pip install kaggle_environments
"""
from kaggle_environments import make, evaluate

# create game env
# set debug=True to see errors

env = make("connectx", debug=True)

print(list(env.agents))
# ['random', 'negamax']

# two random agents play one game round
env.run(["random", "random"])

# show the game
env.render(mode="ipython")



## agent: python function taking: obs and config
## returns integer with selected column [zero indexing]
## so 0-6 inclusive

# selects random valid column
def agent_random(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.bard[col] == 0]
    return random.choice(valid_moves)

# this agent always choose middle col
def agent_middle(obs, config):
    return config.columns // 2

# this agent always leftmost
def agent_leftmost(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.bard[col] == 0]
    return valide_moves[0]

# obs
# obs.board - game board (python list with one item for each grid location - rowwise)
# obs.mark  - piece assigned to the agent (either 1 or 2)

# config
# config.columns - number of cols in game board (7 for connect four)
# config.rows    - number of rows in game board (6 for connect four)
# config.inarow  - number of pieces a player needs to get in a row in order to win
#                  (4 for connect four)

###############################
# evaluating agents
###############################

# agents play onr game round
env.run([agent_leftmost, agent_random])
# show the game
env.render(mode="ipython")


# better evaluation function

def get_win_percentages(agent1, agent2, n_rounds=100):
    # use default connect four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # agent 1 goes first (roughly half the time)
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds // 2)
    # agent 2 goes first (roughly half the time)
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds - n_rounds//2)]
    print(f"Agent1 Win Percentage: {np.round(outcomes.count([1, -1])/len(outcomes), 2)}")
    print(f"Agent1 Win Percentage: {np.round(outcomes.count([1, 1])/len(outcomes), 2)}")
    print(f"Number of Invalid Plays by Agent 1: {outcomes.count([None, 0])}")
    print(f"Number of Invalid Plays by Agent 2: {outcomes.count([0, None])}")

get_win_percentages(agent1=agent_middle, agent2=agent_random)
get_win_percentages(agent1=agent_leftmost, agent2=agent_random)

# agent_leftmost seems to perform best!!

# https://www.kaggle.com/alexisbcook/one-step-lookahead
#################################

################################
# one step look-ahead
################################

# adding some simple heuristics

"""
1000000 points if agent has four discs in a row (agent won)
1 point agent filled 3 spots and remaining spots is empty
-100 point if opponent filled 3 spots an remaingin spot is empty
"""


# calculate score if agent drops piece in selected column
def score_move(grid, col, mark, config):
    next_grid = drop_piece(grid, col, mark, config)
    score = get_heuristic(next_grid, mark, config)
    return score

# get board at next step if agent drops piece in selected col
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid

# score_move: calculate value of heuristic for grid
def get_heuristic(grid, mark, config):
    num_threes = count_windows(grid, 3, mark, config)
    num_fours  = count_windows(grid, 4, mark, config)
    num_threes_opp = count_windows(grid, 3, mark%2+1, config) 
    score = num_trees - 100*num_threes_opp + 1000000*num_fours
    return score

def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

def count_windows(grid, num_dists, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col+config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col.col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(gird[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows

# the agent is always implemented as a python function
# that accepts two args: obs and config

def agent(obs, config):
    # get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # convert board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # use heuristic to assign a score to each possible board in next turn
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config) for col in valid_moves]))
    # get list of cols (moves) that maximize heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    return random.choice(max_cols)



from kaggle_environments import make, evaluate
# create game env
env = make("connectx")
# agent plays against random agent
env.run([agent, "random"])
# show game
env.render(mode="ipython")

get_win_percentages(agent1=agent, agent2="random") # 98% against 2% win!




####################################
# minimax algorithm N-step lookahead
####################################

"""
1000000 points if agent has 4 discs in a row
-10000 points if opponent has four discs in a row
1 poin if agent filled 3 spots
-100 points if opponed filled 3 spots
"""

# I will skip this



#####################################
# deep RL
#####################################

# instead of Q-table we use
# an MLP/ANN
# input: board (list)
# output: next move col 0 - 6

"""
after each move, agent gets a reward:

IF agent wins game in that move:  +1  reward
ELSE IF agent plays invalid move: -10 reward
ELSE IF opponent wins:            -1  reward
ELSE:                             1/42 reward

# at end of game, agent adds up its reward -> cumulative reward
# game lasted 8 moves 3*(1/42)+1
# 11 moves (opponent went first, so agent 5 times) and opponent won:
#                      4*(1/42)-1
# draw after 21 moves  21*(1/42)
# game lasted 7 moves, ended because agent selecting invalid move
#                       3*(1/42) - 10

goal: find weights of NN that (on average) maximiaze agent's cumulative reward.

the idea of using reward to track the performance of an agent
is core ide of RL.
once we define problem in this way, we can use any of a variety of RL algos
 to produce an agent.

RL

DQN
A2C
PPO

- all weights initially random
- agent plays game, continually tries out new values for weights
  to see how cumulative reward is affected on average
  after many times algo converges
- algo tries to win the game (final reward +1)
       avids -1 and -10
       tries to make game last as long as possible (1/42)
Idea of 1/42 bonus is help alog to converge better
'temporal credit assignment problem'
'reward shaping'

PPO proximal policy optimization algo to create an agent

Stable Baselines is not yet compatible with TensorFlow 2.0.
so we begin by downgreading to TensorFlow 1.0
"""

import tensorflow as tf
tf.__version__
# 1.15.0

# environment has to be made compatible with Stable Baselines
# for that: ConnectFourGym class below
# it implements ConnectX as an OpenAI Gym env
# that uses several methods:

"""
reset() - returns starting random board as 2d numpy arry 6 rows 7 cols
change_reward() - customizeds rewards that agent receives
step(action) - play agent's action choice
               it returns: resulting game board as np array
                           agent's reward of most recent move (+1 -10, -1, 1/42)
                           done - game ended?

how to define envs:
https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html

"""


#################################
# env
#################################

from kaggle_environments import make, evaluate
from gym import spaces

class ConnectFourGym:
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2,shape=(self.rows, self.column, 1), dtype=np.in)
        # tuple corresponding to min max possible rewards
        self.reward_range = (-10 1)
        #StableBaselines throws error if these not defined
        self.spec = None
        self.metadata = None
    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(self.rows, self.columns, 1)
    def change_reward(self, old_reward, done):
        if old_reward == 1: # the agent won the game
            return 1
        elif done: # the opponent won
            return -1
        else: # reward 1/42
            return 1/(self.rows*self.columns)
    def step(self, action):
        # check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid: # play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else: # end the game and penalize agent
            reward, done, _ = -10, True, {}
        return np.array(self.obs['board']).reshape(self.rows, self.colums, 1), reward, done, _
        




# create connectfour environment
env = ConnectFourGym(agent2="random")

# stable baselines requires us to work with "vectorized" environments.
# for this, we can use the DummyVecEnv class

# the Monitor class lets us watch how the agent's performance gradually improves
# as it plays more and more games

"""
!pip install stable_baselines

"""

import os
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

# create directory for logging training info
log_dir = "ppo/"
os.makedirs(log_dir, exist_ok=True)

# logging progress
monitor_env = Monitor(env, log_dir, allow_early_resets=True)

# create vectorized environment
vec_env = DummyVecEnv([lambda: monitor_env])

# the next step is to specify the architecture of the NN
# here a CNN
# more about specify architectures with StableBaselines 
# https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html

# PPO algorithm, our network will output some additional info
# "value" of the input.
# read about "actor-critic networks" if curious.



from stable_baselines import PPO1
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.policies import CnnPolicy

# NN for predicting action values
def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 
                         "c1", 
                         n_filters=32, 
                         filter_size=3, 
                         stride=1,
                         init_scale=np.sqrt(2),
                         **kwargs))
    layer_2 = activ(conv(layer_1,
                         'c2',
                         n_filters=64,
                         filter_size=3,
                         stride=1,
                         init_scale=np.squrt(2),
                         **kwargs))
    layer_2 = conv_to_fc(layer_2)
    return activ(linear(layer_2,
                        'fc1',
                        n_hidden=512,
                        init_scale=np.sqrt(2)))

class CustomCnnPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn)

# initialize agent
model = PPO1(CustomCnnPolicy, vec_env, verbose=0)

"""
weights are set randomly

'train agent' means find weight so NN likely to result in agent selecting good moves.
"""

"""
# Train agent
"""

model.learn(total_timesteps=100000)

# plot cumulative reward
with open(os.path.join(log_dir, "monitor.csv"), 'rt') as fh:
    firstline = fh.readline()
    assert firstline[0] == "#"
    df = pd.read_csv(fh, idnex_col=None)['r']

df.rolling(window=1000).mean().plot()
plt.show()

# finally, specify the trained agent in format required for competition
def agent1(obs, config):
    # use best model to select a column
    col, _ = model.predict(np.array(obs['board']).reshape(config.rows, config.columns, 1))
    # check if selected column is valid
    is_valid = (obs['borad'][int(col)] == 0)
    # if not valid, select random move
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])

# see outcome of one game round against random agent



# create game env
env = make("connectx")

# two random agents play one game round
env.run([agent1, "random"])

# show the game
env.render(mode="ipython")

get_win_percentages(agent1=agent1, agent2="random")
# Agent 1 Win Percentage: 0.73
# Agent 2 Win Percentage: 0.27
# Number of Invalid Plays by Agent 1: 0
# Number of Invalid Plays by Agent 2: 0

# only trained to beat "random"
# self-play
# https://openai.com/blog/competitive-self-play/
# will give better results

# free further
# https://www.youtube.com/watch?v=2pWv7GOvuf0
# http://www.incompleteideas.net/book/RLbook2018.pdf
# https://github.com/dennybritz/reinforcement-learning
# https://sites.google.com/corp/view/deep-rl-bootcamp/lectures








################################################################################

# https://www.intel.com/content/www/us/en/artificial-intelligence/posts/demystifying-deep-reinforcement-learning.html?wapkw=demystifying%20deep%20reinforcement%20learning







intro: https://github.com/keon/deep-q-learning

code explained: https://keon.io/deep-q-learning/

keon has interesting 
blogs

also korean book
https://github.com/keon/3-min-pytorch

especially algorithms - python3
# there the graph section is super interesting!!
https://github.com/keon/algorithms

and awesome nlp
https://github.com/keon/awesome-nlp











################################################################################

################################################
# creating gym environment
################################################

# https://www.novatec-gmbh.de/en/blog/creating-a-gym-environment/




################################
# installing gym
################################

"""
git clone https://github.com/openai/gym
cd gym pip install -e .

# for this special calse we need pygame too, since bubble shooter is based
# on it (you can skip this in other cases)

python3 -m pip install -U pygame --user
"""


#################################
# setup package structure
#################################

"""
gym environments come in pip package structure

mkdir -p gym-bubbleshooter/gym_bubbleshooter/env
touch gym-bubbleshooter/README.md # can contain short description of your env

nano gym-bubbleshooter/setup.py

'''
m setuptools import setup

setup(name='gym_bubbleshooter,
      version='0.1',
      install_requires=['gym', 'numpy', 'pygame']
)
'''

nano gym-bubbleshooter/gym_bubbleshooter/__init__.py

'''
from gym.envs.registration import register

register(id='BubbleShooter-v0', 
    entry_point='gym_bubbleshooter.envs:BubbleShooterEnv', 
)
'''

nano gym-bubbleshooter/gym_bubbleshooter/envs/__init__.py

'''
m gym_bubbleshooter.envs.bubbleshooter_env import BubbleShooterEnv
'''

core of the env is:
gym-bubbleshooter/gym_bubbleshooter/envs/bubbleshooter_env.py

nano gym-bubbleshooter/gym_bubbleshooter/envs/bubbleshooter_env.py

4 methods are needed for an gym env:
1. __init__(): initialize class and set initial values
2. step(): take steps and actions -> step env 1 step ahead
   returning: - the next state
              - the reward for that action
              - boolean (episode is over?)
              - and some other info
3. reset(): reset state and return random state
4. render(): visualize state of env in some form

'''
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class BubleShooterEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        pass
    def step(self, action):
        pass
    def reset(self):
        pass
    def render(self, mode='human', close=False):
        pass

'''

separate the game logic from rendering, so that you can interact with
the game using 'step()' instead of keyboard

whole collision logic had to be rewritten

look at other gym envs to get a feeling for how they work and their usage of spaces

# after implementing, install the env
pip install -e .
"""

port gym
import gym_bubbleshooter
env = gym.make('bubbleshooter-v0')



################################
# creating own environments
################################

# https://gym.openai.com/docs/


# https://www.novatec-gmbh.de/en/blog/deep-q-networks


"""

taxi
https://www.novatec-gmbh.de/en/blog/introduction-to-q-learning

env
https://www.novatec-gmbh.de/en/blog/deep-q-networks

customized env
https://www.novatec-gmbh.de/en/blog/creating-a-gym-environment/


# observation-space: 10^174

# CartPole-v1 env


# implement the agent

agent.train()

initialize class by creating env and adopting all its specific envs
ignore memory and batchsize for now (required for experience replay)
gamma: discount-factor
learning-rate with decay need for optimize of our DNN
win_threshold - determines whether and env is solved

for carpole game agent has to score ~195 over 100 consecutive trials
taxi game is solved if value exceeds 9.7

"""


import gym
from collections import deque

# model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np

import random

class DQNAgent():
    def __init__(self, env_id, path, episodes,
                 max_env_steps,
                 win_threshold,
                 epsilon_decay,
                 state_size=None,
                 action_size=None,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 gamma=1,
                 alpha=.01,
                 alpha_decay=.01,
                 batch_size=16,
                 prints=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_id)
        
        if state_size is None:
            self.state_size = self.env.observation_space.n
        else:
            self.state_size = state_size
        
        if action_size is None:
            self.action_size = self.env.action_space.n
        else:
            self.action_size = action_size
        
        self.episodes = episodes
        self.env._max_episode_steps = max_env_steps
        self.win_threshold = win_threshold
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.path = path                 # model_fpath
        self.prints = prints             # print_his_scores?
        self.model = self._build_model()


    # model creator
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="tanh"))
        model.add(Dense(48, activation="tanh"))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mse",
                      optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model
    # act method iither random action or use model to predict highest Q-value
    def act(self, state):
        if (np.random.ranom() <= self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size)
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = rward if done else reward + self.gamm * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *=self.epsilon_decay
    

















