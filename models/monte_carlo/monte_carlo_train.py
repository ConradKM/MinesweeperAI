import gym
import time
import numpy as np
import gym_minesweeper
import copy
import pickle
import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

NUM_OF_MINES = 1
SIZE = 3
EPISODES = 100000
all_moves = {}

def monte_carlo_best_move(obs, env, epilson = 0.8):
    obs_key = tuple(map(tuple, obs))
    valid_moves, p_moves = get_valid_moves(obs)

    if obs_key in all_moves:
        p_moves = all_moves[obs_key]

    best_move = None
    best_reward = -np.inf

    for move in valid_moves:

        simulation_env = env.copy()
        simulation_obs, reward, done, info = simulation_env.step(move)

        if reward > best_reward:
            best_move = move
            best_reward = reward
        
        p_moves[move] += reward

        # print('-'*10)
        # print("MOVE " + str(move))
        # print("Move Reward:", str(reward))
        # print("Best: ", str(best_reward), str(best_move))

    # print(p_moves)
    
    all_moves[obs_key] = p_moves
    if epilson > random.random():
        best_move = random.choice(valid_moves)
    
    return best_move

def convert_to_probabilities(all_moves):
    probabilities = {}

    for obs_key, p_moves in all_moves.items():
        rewards = np.array(p_moves, copy=True)  # 3x3 array

        # Treat -10 or np.nan as invalid (you can adjust threshold here)
        rewards[(rewards == -10) | np.isnan(rewards)] = np.nan

        # Mask valid entries
        valid_mask = ~np.isnan(rewards)
        valid_rewards = rewards[valid_mask]

        if valid_rewards.size == 0:
            probs = np.full((3, 3), np.nan)  # No valid moves
        else:
            # Shift rewards to be positive so we can normalize even with negatives
            min_reward = np.min(valid_rewards)
            shifted = valid_rewards - min_reward + 1e-6  # avoid zero-division

            # Normalize
            normalized = shifted / np.sum(shifted)

            # Fill back into 3x3 array
            probs = np.full((3, 3), np.nan)
            probs[valid_mask] = normalized

        probabilities[obs_key] = probs

    return probabilities  

def get_valid_moves(obs):
    moves = []
    p_moves = np.full((SIZE,SIZE),np.nan)

    for row in range(obs.shape[0]):
        for col in range(obs.shape[1]):
            if obs[row, col] == -1:
                moves.append((row, col))
                p_moves[row,col] = 0 # Valid mvoe
    
    return moves, p_moves

old_count = 0
for num_of_mines in range(1,9):
    env = gym.make("Minesweeper-v0", height=SIZE, width=SIZE, num_mines=num_of_mines, reveal_surrounding=False)
    print("Minesweeper Simulation [ Num of Mines:",num_of_mines,"]")
    for episode in range(int(EPISODES/9)):
        obs = env.reset()
        done = False
        while not done:
            # env.render()
            action = monte_carlo_best_move(obs,env)
            
            obs, reward, done, info = env.step(action)
            
            # print(f"Action: {action}, Reward: {reward}")
            
            # print(np.array(obs), "\n" + "-"*40)
            # if len(get_valid_moves(obs)) == NUM_OF_MINES:
            #     print("WIN")
    env.close()
    print("Number of 3x3 Observations added: ", len(all_moves) - old_count)
    old_count = len(all_moves)


# for num_of_mines in range(1,9):
#     env = gym.make("Minesweeper-v0", height=SIZE, width=SIZE, num_mines=num_of_mines, reveal_surrounding=True)
#     for episode in range(3333):
#         obs = env.reset()
#         done = False
#         while not done:
#             env.render()
#             action = monte_carlo_best_move(obs,env)
            
#             obs, reward, done, info = env.step(action)
            
#             # print(f"Action: {action}, Reward: {reward}")
            
#             # print(np.array(obs), "\n" + "-"*40)
#             # if len(get_valid_moves(obs)) == NUM_OF_MINES:
#             #     print("WIN")
#     env.close()

all_moves = convert_to_probabilities(all_moves)


# for obs_key, p_moves in all_moves.items():
#         print("Observation:")
#         obs_array = np.array(obs_key)
#         print(obs_array)
#         print("\nAssociated p_moves (Total Rewards):")
#         print(np.array2string(p_moves, formatter={'float_kind': lambda x: f"{x:6.4f}"}))
#         print("-" * 50)


filename= str('models/monte_carlo/pickles/3x3model.pkl')
with open(filename, 'wb') as f:
        pickle.dump(all_moves, f)
        print(f"Model saved to {filename}")
