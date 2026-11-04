import os
import pandas as pd

FILES = [
    ("results/results_re40000.csv",       "3-number wings, Re 40,000",         "new"),
    ("results/results_re20000.csv",       "the same wings again, Re 20,000",   "again"),
    ("results/results_round2_re20000.csv", "300 thin wings, Re 20,000",         "new"),
    ("results/cst_data.csv",              "8-number wings, big run, Re 20,000", "new"),
    ("results/rep1.csv",                  "8-number wings, repeat 1",          "new"),
    ("results/rep2.csv",                  "8-number wings, repeat 2",          "new"),
    ("results/rep3.csv",                  "8-number wings, repeat 3",          "new"),
]

print(f"{'file':38s}{'XFOIL runs':>11s}{'good':>8s}")
runs = 0
shapes = 0
good_total = 0
for path, label, kind in FILES:
    if not os.path.exists(path):
        print(f"{label:38s}   (file not found: {path})")
        continue
    df = pd.read_csv(path)
    good = int(((df["n_ok"] >= 10) & df["best_LD"].notna()).sum())
    print(f"{label:38s}{len(df):11d}{good:8d}")
    runs += len(df)
    good_total += good
    if kind == "new":
        shapes += len(df)

print()
print("Total XFOIL runs (wing tests):", runs)
print("Different wing shapes        :", shapes)
print("Runs that passed the checks  :", good_total)