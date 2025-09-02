import numpy as np
import random
import torch
import gym_minesweeper
import gym
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SIZE = 5
NUM_OF_MINES = 2 
RENDERING = False
EPISODES = 100000
PERIODIC_SAVE = True

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

class DQN(nn.Module):
    def __init__(self, height, width, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * height * width, num_actions)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (B, 1, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DQNAgent: 
    def __init__(self, env, buffer_size=5000, batch_size=128, gamma=0.9, lr=1e-4, target_update_freq=10):
        self.env = env
        self.height = env.height
        self.width = env.width
        self.num_actions = self.height * self.width
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(torch.backends.mps.is_available())  
        print(torch.backends.mps.is_built())  

        self.policy_net = DQN(self.height, self.width, self.num_actions).to(self.device)
        self.target_net = DQN(self.height, self.width, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.total_episodes = EPISODES  # Automatically syncs with main config
        self.eps_k = -np.log((self.eps_end + 1e-3) / self.eps_start) / self.total_episodes
        self.steps_done = 0
        self.episode = 0

    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.eps_k * self.episode)
        self.steps_done += 1

        # Get valid actions (unrevealed tiles)
        valid_actions = [(i, j) for i in range(self.height) for j in range(self.width) if state[i][j] == -1]
        
        if not valid_actions:
            # All tiles revealed or no valid move — fallback
            return (0, 0)

        if random.random() < eps_threshold:
            return random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)

            # Convert q_values to (x, y)
            flat_indices = [a[0]*self.width + a[1] for a in valid_actions]
            valid_qs = q_values.view(-1)[flat_indices]
            best_idx = torch.argmax(valid_qs).item()
            return valid_actions[best_idx]

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(
            [a[0] * self.width + a[1] for a in batch.action], dtype=torch.long
        ).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        if self.episode % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath="models/deep_q/dqn.pth"):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode': self.episode
        }, filepath)
        print(f"✅ Model saved to {filepath}")

def train(env, agent, num_episodes=300):
    rewards = []
    averaged_rewards = []
    wins = 0
    winning_games = []
    steps = 0

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        agent.episode = ep
        random_move = 0

        done = True
        while done:
        
            state, reward, done, info = env.step(env.action_space.sample())
            if done:
                state = env.reset()
        done = False

        total_reward += reward

        while not done:
            if RENDERING:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if reward == -1:
                random_move += 1
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.optimize_model()
            if done and env.win_reward == reward:
                print(f"Episode {ep+1}: WIN")
                wins += 1
                winning_games.append(ep+1)
                



        agent.update_target_network()
        rewards.append(total_reward)
        if len(rewards) >= EPISODES / 1000:
            if ((ep+1) % (EPISODES / 1000)) == 0:
                averaged_rewards.append(np.mean(rewards))
                if((ep+1) % (EPISODES / 10)) == 0 and PERIODIC_SAVE:
                    agent.save()
                    print(f"Agent saved at {ep+1}")
            rewards.pop(0)
        if ep % 1000 == 0:  # print every 1000 episodes
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor).squeeze(0).cpu().numpy()
                print(f"Q-values (flattened): {q_values.round(2)}")


        recent_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
    
        print(f"Episode {ep+1}/{num_episodes} | Total Reward: {total_reward} | Moving Avg (10): {recent_avg:.1f} | Steps: {agent.steps_done - steps} | Random Moves: {random_move}")
        steps = agent.steps_done

    print(wins)
    print(winning_games)

    return averaged_rewards


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

if __name__ == "__main__":
    env = gym.make("Minesweeper-v0", height=SIZE, width=SIZE, num_mines=NUM_OF_MINES)
    agent = DQNAgent(env)
    rewards = train(env, agent, num_episodes=EPISODES)
    averaged_rewards = moving_average(rewards, window_size=10)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Total Reward (raw)")
    plt.plot(range(len(averaged_rewards)), averaged_rewards, label="Moving Average (window=10)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress: Reward per Episode")
    plt.legend()
    plt.grid()
    plt.show()

    agent.save()