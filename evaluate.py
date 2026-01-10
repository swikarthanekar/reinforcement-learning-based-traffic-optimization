import numpy as np
import torch
import matplotlib.pyplot as plt
from env.traffic_env import TrafficEnv
from agent.dqn import DQN
import os

STATE_DIM = 8
ACTION_DIM = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_fixed_timer(env, switch_interval=10):
    state = env.reset()
    total_wait = 0
    light_state = 0

    for t in range(env.max_steps):
        if t % switch_interval == 0:
            light_state = 1 - light_state

        action = light_state 
        state, reward, done, _ = env.step(action)
        total_wait += -reward

        if done:
            break

    return total_wait / env.max_steps


def run_dqn(env, model):
    state = env.reset()
    total_wait = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        state, reward, done, _ = env.step(action)
        total_wait += -reward

    return total_wait / env.max_steps


if __name__ == "__main__":
    densities = [0.1, 0.3, 0.5]
    fixed_results = []
    rl_results = []

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "dqn_traffic.pth")

    model = DQN(STATE_DIM, ACTION_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    for d in densities:
        fixed_runs = []
        rl_runs = []

        for _ in range(5):
            env = TrafficEnv(arrival_rate=d)
            fixed_runs.append(run_fixed_timer(env))

            env = TrafficEnv(arrival_rate=d)
            rl_runs.append(run_dqn(env, model))

        fixed_results.append(np.mean(fixed_runs))
        rl_results.append(np.mean(rl_runs))

    plot_path = os.path.join(RESULTS_DIR, "wait_time_vs_density.png")

    plt.figure()
    plt.plot(densities, fixed_results, marker="o", label="Fixed Timer")
    plt.plot(densities, rl_results, marker="o", label="DQN Agent")
    plt.xlabel("Traffic Density")
    plt.ylabel("Average Waiting Time")
    plt.title("Fixed Timer vs RL Traffic Signal Control")
    plt.legend()
    plt.grid(True)

    plt.savefig(plot_path)
    plt.show()

    print(f"Evaluation complete. Plot saved to:\n{plot_path}")
