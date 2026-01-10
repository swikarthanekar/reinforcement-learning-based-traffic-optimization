import numpy as np
import random

class TrafficEnv:
    def __init__(self, max_steps=200, arrival_rate=0.3):
        self.lanes = ["N", "S", "E", "W"]
        self.max_steps = max_steps
        self.arrival_rate = arrival_rate  
        self.reset()

    def reset(self):
        self.step_count = 0

        self.cars = {lane: 0 for lane in self.lanes}

        self.waiting_time = {lane: 0 for lane in self.lanes}

        self.light_state = 0

        self.ambulance_lane = None

        return self._get_state()

    def _spawn_cars(self):
        for lane in self.lanes:
            if random.random() < self.arrival_rate:
                self.cars[lane] += 1

    def _update_waiting_time(self):
        for lane in self.lanes:
            self.waiting_time[lane] += self.cars[lane]

    def _move_cars(self):
        if self.light_state == 0:  
            green_lanes = ["N", "S"]
        else:  
            green_lanes = ["E", "W"]

        if self.ambulance_lane:
            green_lanes = [self.ambulance_lane]

        for lane in green_lanes:
            if self.cars[lane] > 0:
                self.cars[lane] -= 1

    def _maybe_spawn_ambulance(self):
        if random.random() < 0.02:
            self.ambulance_lane = random.choice(self.lanes)
        else:
            self.ambulance_lane = None

    def _get_state(self):
        state = []
        for lane in self.lanes:
            state.append(self.cars[lane])
        for lane in self.lanes:
            state.append(self.waiting_time[lane])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        Actions:
        0 -> NS green
        1 -> EW green
        2 -> switch light
        """

        self.step_count += 1

        if action == 0:
            self.light_state = 0
        elif action == 1:
            self.light_state = 1
        elif action == 2:
            self.light_state = 1 - self.light_state

        self._maybe_spawn_ambulance()
        self._spawn_cars()
        self._move_cars()
        self._update_waiting_time()

        total_wait = sum(self.waiting_time.values())
        reward = -total_wait

        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done, {
            "cars": self.cars,
            "ambulance_lane": self.ambulance_lane
        }
