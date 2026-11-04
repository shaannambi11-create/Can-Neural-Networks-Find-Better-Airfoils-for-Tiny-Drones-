import pandas as pd

df = pd.read_csv("results/cst_data.csv")
good = df[(df["n_ok"] >= 10) & df["best_LD"].notna()]

print("How many wings passed the filter:")
for src in ["random", "ai", "random_arm"]:
    tested = df[df["source"] == src]
    passed = good[good["source"] == src]
    if len(tested) > 0:
        pct = 100 * len(passed) / len(tested)
        print(f"   {src:11s} tested {len(tested):4d}   passed {len(passed):4d}   ({pct:.0f}%)")

ai = good[good["source"].isin(["random", "ai"])].sort_values("order")
rd = good[good["source"].isin(["random", "random_arm"])].sort_values("order")
n = min(len(ai), len(rd))
print()
print("Equal number of good wings for each side:", n)
print("Best score, AI-guided :", ai["best_LD"].head(n).max())
print("Best score, random    :", rd["best_LD"].head(n).max())