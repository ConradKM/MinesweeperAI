# Gym Minesweeper
**Gym Minesweeper is an environment for OpenAI Gym simulating a minesweeper game.** [PyPI gym-minesweeper](https://pypi.org/project/gym-minesweeper/)

<p align="center">
<img align="center" src="https://jeffreyyao.github.io/images/minesweeper_solver.gif"/>
</p>

---

## Installation

```bash
pip install gym-minesweeper
```

## Running

```python
import gym
import gym_minesweeper

env = gym.make("Minesweeper-v0") # 16x16 map with 40 mines
env.reset()

done = False
while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

env.close()
```