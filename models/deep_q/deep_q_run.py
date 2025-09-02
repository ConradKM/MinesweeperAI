import torch
import gym
import gym_minesweeper
import numpy as np
from deep_q import DQN, DQNAgent
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SIZE = 8
NUM_OF_MINES = int(SIZE**2 * 0.075)
EPISODES = 1000
RENDERING = True


def load_agent(env, filepath="models/deep_q/models/8x8_10_100000.pth"):
    agent = DQNAgent(env)
    checkpoint = torch.load(filepath, map_location=agent.device)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.steps_done = checkpoint['steps_done']
    agent.episode = checkpoint['episode']
    agent.epsilon = 0.0
    print(f"✅ Loaded model from {filepath}")
    return agent

def evaluate_agent(agent, env, num_episodes=EPISODES):
    wins = 0
    all_rewards = 0
    all_uncovered = 0
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if RENDERING:
                env.render()
            action = agent.select_action(state)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                all_rewards += total_reward
                all_uncovered += np.sum(obs != -1)
                if reward == env.win_reward:
                    wins += 1
            time.sleep(0.5)

        # print(f"Episode {ep+1}/{num_episodes} | Total Reward: {total_reward:.2f}")



    print(f"✅ Evaluation complete:")
    print(f"Win rate: {wins/EPISODES}")
    print(f"Average Reward per Episode: {all_rewards / EPISODES}")
    print(f"Average % of board revealed: {(all_uncovered) / (SIZE * SIZE * EPISODES)}")
    

env = gym.make("Minesweeper-v0", height=SIZE, width=SIZE, num_mines=NUM_OF_MINES)
agent = load_agent(env)
print(f"Running {SIZE}x{SIZE} board with {NUM_OF_MINES} mines DQN for {EPISODES} episodes")
evaluate_agent(agent, env)