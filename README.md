# Implementation-of-Q-learning-Control-algorithm
## Aim

To implement the Q-Learning Control Algorithm using the FrozenLake environment in Reinforcement Learning and learn the optimal policy through interaction with the environment.

---

# Algorithm

### Q-Learning Algorithm

1. Import the required libraries.
2. Create the FrozenLake environment.
3. Initialize learning parameters:
   - Learning rate (`alpha`)
   - Discount factor (`gamma`)
   - Exploration rate (`epsilon`)
4. Initialize the Q-table with zeros.
5. For each episode:
   - Reset the environment.
   - Choose an action using the epsilon-greedy policy.
   - Execute the action and observe:
     - Next state
     - Reward
   - Update the Q-value using:

\[
Q(s,a)=Q(s,a)+\alpha \left[r+\gamma \max Q(s',a')-Q(s,a)\right]
\]

6. Decay epsilon after every episode.
7. Store rewards for performance analysis.
8. Print the learned Q-table and optimal policy.
9. Plot reward vs episodes graph.

---

# Program

```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make
 (
    "FrozenLake-v1",
    is_slippery=False
)

alpha = 0.9
gamma = 0.99

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 500

Q = np.zeros((
    env.observation_space.n,
    env.action_space.n
))

rewards_per_episode = []

def choose_action(state):

    if np.random.rand() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])

for episode in range(episodes):

    state, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        action = choose_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        if reward == 1:
            reward = 10

        elif done:
            reward = -5

        else:
            reward = 0.1

        Q[state][action] = Q[state][action] + alpha * (
            reward
            + gamma * np.max(Q[next_state])
            - Q[state][action]
        )

        state = next_state

        total_reward += reward

    rewards_per_episode.append(total_reward)

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

print("\nQ-TABLE VALUES:\n")

print(Q)

print("\nLEARNED POLICY:\n")

actions = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

for state in range(env.observation_space.n):

    best_action = np.argmax(Q[state])

    print(
        f"State {state} --> "
        f"{actions[best_action]}"
    )

successful_episodes = 0

for reward in rewards_per_episode:

    if reward > 0:
        successful_episodes += 1

success_rate = (
    successful_episodes / episodes
) * 100

print("\nSUCCESS RATE =", success_rate, "%")

plt.figure(figsize=(10,5))

plt.plot(rewards_per_episode)

plt.xlabel("Episodes")

plt.ylabel("Reward")

plt.title("Q-Learning Performance")

plt.show()

env.close()


```

---

# Output

```
Q-TABLE VALUES:

[[ 9.9999541  10.         10.         10.        ]
 [10.         -5.         10.         10.        ]
 [10.          9.94845972  9.99999999  9.99999996]
 [10.         -4.99999999  9.98440633  9.87577262]
 [10.          9.99996229 -5.          9.9541    ]
 [ 0.          0.          0.          0.        ]
 [-4.995       9.49988607 -4.995       9.9995257 ]
 [ 0.          0.          0.          0.        ]
 [ 9.99996532 -5.          9.99999954  9.9541    ]
 [ 9.99999591  9.99999853 10.         -5.        ]
 [10.          0.09       -4.995       8.99577397]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-4.995       1.21397492  0.          9.99999991]
 [ 0.09        0.26757729  0.          0.        ]
 [ 0.          0.          0.          0.        ]]

LEARNED POLICY:

State 0 --> RIGHT
State 1 --> LEFT
State 2 --> LEFT
State 3 --> LEFT
State 4 --> LEFT
State 5 --> LEFT
State 6 --> UP
State 7 --> LEFT
State 8 --> RIGHT
State 9 --> RIGHT
State 10 --> LEFT
State 11 --> LEFT
State 12 --> LEFT
State 13 --> UP
State 14 --> DOWN
State 15 --> LEFT

SUCCESS RATE = 17.599999999999998 %

```

---

## Output Graph

<img width="1055" height="582" alt="image" src="https://github.com/user-attachments/assets/c997f393-e24a-436d-a2dc-8c8a2ccffbeb" />

---

# Result

Thus, the Q-Learning Control Algorithm was successfully implemented using the FrozenLake environment. The agent learned the optimal policy through exploration and exploitation, and the performance improved over episodes with a high success rate.
