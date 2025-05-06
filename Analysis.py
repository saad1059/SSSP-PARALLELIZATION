import csv
import matplotlib.pyplot as plt
from collections import defaultdict

# Read the CSV data
performance = defaultdict(list)

with open("performance.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 3:
            continue
        impl_type, dataset, time = row
        performance[dataset].append((impl_type, float(time)))

# Plot
for dataset, entries in performance.items():
    types = [entry[0] for entry in entries]
    times = [entry[1] for entry in entries]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(types, times, color='skyblue')
    
    # Add labels on top of bars
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{time:.3f}s",
                 ha='center', va='bottom')

    plt.title(f"Execution Time for Dataset: {dataset}")
    plt.xlabel("Implementation Type")
    plt.ylabel("Time (seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"performance_{dataset}.png")
    plt.show()