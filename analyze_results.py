import argparse
import json
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd

def is_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def analyze_results(data):
    results = data.get("results", {})
    rows = []

    for task, metrics in results.items():
        if task.startswith("mmlu_") and task != "mmlu":
            continue
        alias = metrics.get("alias", task).strip()
        acc = metrics.get("acc,none", None)
        em_strict = metrics.get("exact_match,strict-match", None)

        rows.append([
            alias,
            float(acc) if is_number(acc) else None,
            float(em_strict) if is_number(em_strict) else None,
        ])
    return rows

def plot_results(rows):
    tasks = []
    accs = []

    for row in rows:
        task = row[0]
        acc_str = row[1]

        try:
            acc = float(acc_str)
        except (TypeError, ValueError):
            acc = None

        if acc is not None:
            tasks.append(task)
            accs.append(acc)

    plt.figure(figsize=(10, 6))
    y = range(len(tasks))
    plt.barh(y, accs, height=0.4, label="Accuracy", align='center')
    plt.yticks(y, tasks)
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Task")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("accuracy_plot.png")
    print("Plot saved as 'accuracy_plot.png'")


parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, help="path to the results JSON file")
parser.add_argument("--plot", action="store_true", help="plot accuracy metrics")
args = parser.parse_args()

if not os.path.exists(args.path):
    print(f"Error: {args.path} does not exist")

with open(args.path, "r") as f:
    data = json.load(f)

rows = analyze_results(data)
headers = ["Task", "Accuracy", "Exact Match (Strict)"]
print(tabulate(rows, headers=headers, floatfmt=".3f"))

df = pd.DataFrame(rows, columns=["Task", "Accuracy", "Exact Match (Strict)"])
df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
df["Exact Match (Strict)"] = pd.to_numeric(df["Exact Match (Strict)"], errors="coerce")
macro_avg = df["Accuracy"].mean(skipna=True)
print(f"Macro-average accuracy: {macro_avg:.3f}")

if args.plot:
    plot_results(rows)