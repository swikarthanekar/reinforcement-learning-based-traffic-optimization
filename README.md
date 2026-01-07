# 🚦 Autonomous Traffic Signal Optimization using Reinforcement Learning

An AI-based traffic signal controller that dynamically optimizes signal timings using **Deep Reinforcement Learning (DQN)**.  
The system adapts to real-time traffic density, prioritizes emergency vehicles, and significantly reduces average waiting time compared to traditional fixed-timer traffic signals.

---

## 📌 Project Overview

Conventional traffic lights operate on static timers and fail to adapt to changing traffic conditions.  
This project builds an **intelligent traffic controller** that learns optimal signal policies through interaction with a simulated environment.

The controller is evaluated against a fixed-timer baseline to demonstrate measurable performance improvements, especially under high traffic density.

---

## 🎯 Objectives

- Minimize total vehicle waiting time  
- Adapt signal timing based on real-time traffic conditions  
- Handle emergency vehicle (ambulance) priority  
- Compare AI-based control with static traffic signals  

---

## 🧠 Approach

- **Reinforcement Learning Algorithm**: Deep Q-Network (DQN)  
- **Framework**: PyTorch  
- **Environment**: Custom Python-based traffic simulator  
- **Baseline**: Fixed-timer traffic signal controller  

---

## 🛣️ Traffic Environment

- 4-way intersection: North, South, East, West  
- Vehicles arrive stochastically based on traffic density  
- Each lane tracks:
  - Queue length  
  - Cumulative waiting time  
- Traffic lights control vehicle movement per timestep  

---

## 📊 State, Action & Reward

### State Representation
An 8-dimensional state vector:

```
[cN, cS, cE, cW, wN, wS, wE, wW]
```

- `c*` → Number of cars in each lane  
- `w*` → Cumulative waiting time per lane  

### Action Space
- `0` → North–South green  
- `1` → East–West green  
- `2` → Switch signal  

### Reward Function
```
Reward = − (Total cumulative waiting time)
```

---

## 🚑 Emergency Vehicle Handling

- Emergency vehicles are randomly introduced  
- When detected:
  - Corresponding lane is immediately cleared  
  - RL policy is overridden temporarily  

---

## 🧩 Model Architecture

- Input Layer: 8 neurons  
- Hidden Layers:
  - Fully connected (64 units, ReLU)  
  - Fully connected (64 units, ReLU)  
- Output Layer: 3 Q-values  

### Training Stabilization
- Experience Replay Buffer  
- Target Network  
- ε-greedy exploration strategy  

---

## 📁 Project Structure

```
traffic-rl/
│
├── env/
│   └── traffic_env.py
├── agent/
│   └── dqn.py
├── models/
│   └── dqn_traffic.pth
├── results/
│   └── wait_time_vs_density.png
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## 🏋️ Training the Model

```bash
python train.py
```

The trained model is saved to:

```
models/dqn_traffic.pth
```

---

## 📈 Evaluation & Results

The RL controller is evaluated against a fixed-timer baseline under multiple traffic densities.  
Results show consistent reduction in average waiting time, with the largest gains under heavy congestion.

---

## ▶️ Run Evaluation

```bash
python evaluate.py
```

This generates:

```
results/wait_time_vs_density.png
```

---

## 🧰 Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- NumPy  
- PyTorch  
- Matplotlib  

---

## 🔮 Future Improvements

- Multi-intersection coordination  
- Advanced RL algorithms (PPO, A2C)  
- Real-time dashboards  

---

## 👤 Author

**Swikar Thanekar**  
AI & Reinforcement Learning Project (Phase 2)
