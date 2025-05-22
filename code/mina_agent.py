import numpy as np
import pandas as pd

class MiNAAgent:
    def __init__(self, bits=8, w=10, tau_u=0.33, tau_delta=0.05, alpha=0.3, delta=2):
        self.bits = bits
        self.w = w
        self.tau_u = tau_u
        self.tau_delta = tau_delta
        self.alpha = alpha
        self.delta = delta
        self.entropy_history = []
        self.comparator_theta = 0.5  # initial threshold
        self.phi_history = []

    def entropy(self, vec):
        p = np.mean(vec)
        if p == 0 or p == 1: return 0
        return -p*np.log2(p) - (1-p)*np.log2(1-p)

    def act(self, state):
        avg = np.mean(state)
        return 1 if avg > self.comparator_theta else 0

    def update(self, state, t):
        ent = self.entropy(state)
        self.entropy_history.append(ent)
        if len(self.entropy_history) < 2: return False
        delta_u = self.entropy_history[-2] - self.entropy_history[-1]
        if self.entropy_history[-2] > self.tau_u and delta_u > self.tau_delta:
            self.comparator_theta += self.alpha * delta_u
            self.phi_history.append(self.comparator_theta)
            print(f"Noetic event at t={t}: entropy_drop={delta_u:.4f}, theta={self.comparator_theta:.4f}")
            return True
        return False

# --- MAIN EXPERIMENT ---

state = np.random.randint(0, 2, 8)
agent = MiNAAgent()
results = []

for t in range(20):
    action = agent.act(state)
    idx = np.random.randint(0, 8)
    state[idx] = 1 - state[idx]
    is_noet = agent.update(state, t)
    results.append({
        "time": t,
        "entropy": agent.entropy_history[-1],
        "comparator_theta": agent.comparator_theta,
        "noetic_event": int(is_noet)
    })

# SAVE to CSV
df = pd.DataFrame(results)
df.to_csv("../data/experiment_results.csv", index=False)
print("Saved results to data/experiment_results.csv")
