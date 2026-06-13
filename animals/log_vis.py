import json
import pandas as pd
import matplotlib.pyplot as plt

# Load data
records = []
with open("training_metrics.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if "metrics" in d:
            records.append(d["metrics"])

df = pd.DataFrame(records)

# Plot key metrics in a grid
metrics_to_plot = [
    "ep_reward", "ep_length", "avg_int_reward",
    "value_loss_ext", "value_loss_int", "rnd_loss",
    "policy_loss", "entropy", "current_avg_life_force"
]

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, metric in zip(axes.flat, metrics_to_plot):
    ax.plot(df["step"], df[metric])
    ax.set_title(metric)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)