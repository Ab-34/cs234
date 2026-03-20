import csv
import matplotlib.pyplot as plt

# ── File paths ── edit these to point to your CSVs ──────────────────────────
CSV_CURRICULUM = "/home/yash/Stanford/CS234/project/cs234/PPO_55.csv"   # BC + PPO curriculum based
CSV_HEURISTIC  = "/home/yash/Stanford/CS234/project/cs234/PPO_22.csv"    # BC + PPO heuristic based
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    steps, values = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["Step"]))
            values.append(float(row["Value"]))
    return steps, values

cur_steps, cur_values = load_csv(CSV_CURRICULUM)
heu_steps, heu_values = load_csv(CSV_HEURISTIC)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot curriculum-based
ax.plot(cur_steps, cur_values, label="BC + PPO curriculum based", linewidth=2)

# Plot heuristic-based
ax.plot(heu_steps, heu_values, label="BC + PPO heuristic based", linewidth=2)

# Flat baseline at 559
ax.axhline(y=559, color="red", linestyle="--", linewidth=1.8, label="Vanilla BC")

ax.set_xlabel("Iteration", fontsize=13)
ax.set_ylabel("Reward", fontsize=13)
ax.set_title("Reward vs Iteration", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reward_plot.png", dpi=150)
print("Saved reward_plot.png")
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Replace these with your reward values
# method1 = [56, 63, 54, 56, 53]
# method2 = [85, 82, 83, 84, 80]
# method3 = [85, 86, 80, 81, 80]
# method4 = [92, 90, 98, 92, 95]

# data = [method1, method2, method3, method4]

# labels = [
#     "Vanilla BC",
#     "BC + PPO simple heuristic",
#     "BC + PPO VLM based",
#     "BC + PPO ciriculum based"
# ]

# plt.figure(figsize=(8,6))

# box = plt.boxplot(
#     data,
#     labels=labels,
#     patch_artist=True,
#     showmeans=True
# )

# # nicer colors
# colors = ['lightblue','lightgreen','salmon','violet']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)

# plt.ylabel("Success rate (%)")
# plt.title("Comparison of Methods (5 rollouts of 100 episodes)")
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()
