import pandas as pd
import matplotlib.pyplot as plt

# Load your real experiment results
df = pd.read_csv("../data/experiment_results.csv")

# Plot entropy and comparator theta over time
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(df["time"], df["entropy"], label="Entropy", color="blue", marker="o")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Entropy", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(df["time"], df["comparator_theta"], label="Comparator Theta", color="orange", marker="s")
ax2.set_ylabel("Comparator Theta", color="orange")
ax2.tick_params(axis='y', labelcolor="orange")

# Mark noëtic events
for t, is_noet in enumerate(df["noetic_event"]):
    if is_noet:
        ax1.axvline(x=t, color='red', linestyle='--', alpha=0.5)

fig.suptitle("MiNA Experiment Results: Entropy & Comparator Theta Over Time")
fig.tight_layout()
plt.savefig("../figures/experiment_plot.png")
plt.show()
