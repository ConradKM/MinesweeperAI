import gym
import time
import numpy as np
import gym_minesweeper
import copy
import pickle
import random


SIZE = 6
NUM_OF_MINES = 3
EPISODES = 1000
RENDERING = True
random_choice_pick = False
random_choice_amount = 0
all_moves = {}


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def print_obs_with_highlight(obs, center):
    """
    Print the Minesweeper board with the 3x3 region centered at `center` highlighted.
    """
    row_c, col_c = center
    for i in range(obs.shape[0]):
        row_str = ""
        for j in range(obs.shape[1]):
            tile = obs[i, j]
            if abs(i - row_c) <= 1 and abs(j - col_c) <= 1:
                # Highlight this tile (bold yellow)
                row_str += f"\033[1;33m{tile:3}\033[0m"
            else:
                row_str += f"{tile:3}"
        print(row_str)
    print("-" * 40)
    
def get_all_tiles(obs):
    moves = []

    for row in range(1, obs.shape[0] - 1):
        for col in range(1, obs.shape[1] - 1):
            moves.append((row, col))
    
    return moves

def get_valid_moves(obs):
    valid_moves = []
    
    # Iterate through the grid and find all valid unrevealed tiles
    for row in range(obs.shape[0]):
        for col in range(obs.shape[1]):
            if obs[row, col] == -1:  # -1 indicates unrevealed tile
                valid_moves.append((row, col))
    
    return valid_moves

def load_model(filename='models/monte_carlo/pickles/3x3model.pkl'):
    try:
        with open(filename, 'rb') as f:
            all_moves = pickle.load(f)
        print(f"Model loaded from {filename}")
        return all_moves
    except FileNotFoundError:
        print(f"Model file {filename} not found.")
        return {}

def find_best_move(obs):
    best_move = None
    best_probability= -np.inf
    all_tiles = get_all_tiles(obs)
    debug_string = ""
    for middle_tile in all_tiles:
    
        obs3x3 = get_3x3_obs_from_tile(obs, middle_tile)
        obs_key = tuple(map(tuple,obs3x3))
        if obs_key in all_moves:
            move_probability = all_moves[obs_key].copy()  # This is the probability distribution for this move
            debug_string += str(move_probability) + "\n"
            # Find the position of the highest probability move (best move)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    tile = (middle_tile[0] + i, middle_tile[1] + j)
                    if 0 <= tile[0] < obs.shape[0] and 0 <= tile[1] < obs.shape[1]:
                        obs3x3k = get_3x3_obs_from_tile(obs, tile)
                        obs_key = tuple(map(tuple, obs3x3k))
                        if obs_key in all_moves:
                            move_probability_new = all_moves[obs_key]
                            center_prob = move_probability_new[1, 1]
                            old_probability = move_probability[i + 1, j + 1]
                            move_probability[i + 1, j + 1] += center_prob * move_probability[i + 1, j + 1]
                            debug_string += str("New Move Probability for: " + str(i) +" " + str(j) + " : " + str(move_probability[i + 1, j + 1]) + ' = '+ str(center_prob) + " * " + str(old_probability) +"\n")
            debug_string += str(move_probability) + "\n"
            if np.isnan(move_probability).all():
                continue
            max_prob = np.nanmax(move_probability)
            best_position = np.unravel_index(np.nanargmax(move_probability), move_probability.shape)  # Get position of max probability
            
            if max_prob > best_probability:
                best_move_matrix = move_probability
                best_obs = obs3x3
                best_probability = max_prob
                best_move = (middle_tile[0] + best_position[0] - 1, middle_tile[1] + best_position[1] - 1)
                k_middle_tile = middle_tile
                
    if best_move is None:
        # print(np.array(obs3x3))
        best_move = random.choice(get_valid_moves(obs))
        # print("Random Move was made!")
        global random_choice_amount, random_choice_pick
        random_choice_pick = True
        random_choice_amount += 1
        # print(obs)
    # else:   
    #     print("BP: " , best_probability)
    #     print(best_move_matrix,"\n")
    #     print(best_obs)
    #     print_obs_with_highlight(obs,k_middle_tile)
    return best_move, debug_string

    
    
    
def get_3x3_obs_from_tile(obs, tile):
    """
    Extracts a 3x3 observation around a given tile in the grid.

    :param obs: The current grid (2D array).
    :param tile: A tuple (row, col) of the valid tile location.
    :return: A 3x3 numpy array around the valid tile.
    """
    row, col = tile
    
    # Define the boundaries for the 3x3 window
    row_min = max(row - 1, 0)
    row_max = min(row + 2, obs.shape[0])
    
    col_min = max(col - 1, 0)
    col_max = min(col + 2, obs.shape[1])
    
    # Extract the 3x3 window
    return obs[row_min:row_max, col_min:col_max]


all_moves = load_model(str('models/monte_carlo/pickles/3x3model.pkl'))
env = gym.make("Minesweeper-v0", height=SIZE, width=SIZE, num_mines=NUM_OF_MINES)
win_rate = 0
random_choice_loss = 0
all_rewards = 0
all_uncovered = 0
for i in range(EPISODES):
    obs = env.reset()
    done = True
    while done:
        
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            obs = env.reset()
    done = False

    debug_string = ""
    win = False
    total_reward = 0
    while not done:
        debug_string = ""
        random_choice_pick =False
        action, debug_string = find_best_move(obs)
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            all_rewards += total_reward
            all_uncovered += np.sum(obs != -1)
        if len(get_valid_moves(obs)) == NUM_OF_MINES:
            # print("WIN")
            win_rate += 1
            win = True
        if RENDERING:
            env.render()
            time.sleep(2)

    
    # if not win:
    #     # print(debug_string)
    #     if random_choice_pick:
    #         random_choice_loss += 1


print(f"✅ Evaluation complete:")
print(f"Win rate: {win_rate/EPISODES}")
print(f"Average Reward per Episode: {all_rewards / EPISODES}")
print(f"Average % of board revealed: {(all_uncovered) / (SIZE * SIZE * EPISODES)}")
