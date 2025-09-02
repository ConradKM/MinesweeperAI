import gym
import time
import numpy as np
import gym_minesweeper
import copy
import pickle
import random

SIZE = 10
SIZE_SQUARED = SIZE**2
NUM_OF_MINES = 3
print()
EPISODES = 1000
random_choice_pick = False
random_choice_amount = 0
all_moves = {}


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
        
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
        if done:
            all_rewards += total_reward
            all_uncovered += np.sum(obs != -1)
        if reward == env.win_reward:
            win_rate += 1
            win = True



print(f"✅ Evaluation complete:")
print(f"Win rate: {win_rate/EPISODES}")
print(f"Average Reward per Episode: {all_rewards / EPISODES}")
print(f"Average % of board revealed: {(all_uncovered) / (SIZE * SIZE * EPISODES)}")
