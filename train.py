import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from env.traffic_env import TrafficEnv
from agent.dqn import DQN, ReplayBuffer

STATE_DIM = 8
ACTION_DIM = 3
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
NUM_EPISODES = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TrafficEnv(arrival_rate=0.3)
policy_net = DQN(STATE_DIM, ACTION_DIM).to(DEVICE)
target_net = DQN(STATE_DIM, ACTION_DIM).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(capacity=10000)

epsilon = EPSILON_START


for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.randint(ACTION_DIM)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            states = torch.FloatTensor(states).to(DEVICE)
            actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
            next_states = torch.FloatTensor(next_states).to(DEVICE)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

            q_values = policy_net(states).gather(1, actions)

            with torch.no_grad():
                max_next_q = target_net(next_states).max(1, keepdim=True)[0]
                target_q = rewards + GAMMA * max_next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(
        f"Episode {episode+1}/{NUM_EPISODES} | "
        f"Total Reward: {total_reward:.2f} | "
        f"Epsilon: {epsilon:.3f}"
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "dqn_traffic.pth")
torch.save(policy_net.state_dict(), model_path)

print(f"Training complete. Model saved to {model_path}")
