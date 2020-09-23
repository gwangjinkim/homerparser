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























