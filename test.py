import numpy as np
from env.traffic_env import TrafficEnv

env = TrafficEnv()
state = env.reset()

for _ in range(10):
    action = np.random.randint(0, 3)
    state, reward, done, info = env.step(action)
    print(state, reward, info)
