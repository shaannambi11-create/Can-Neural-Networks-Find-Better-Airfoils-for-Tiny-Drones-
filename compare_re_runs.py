import pandas as pd
import matplotlib.pyplot as plt

a = pd.read_csv("results/results_re40000.csv")
b = pd.read_csv("results/results_re20000.csv")
a = a[a["n_ok"] >= 10][["name", "m", "p", "t", "best_LD"]]
a = a.rename(columns={"best_LD": "LD_40k"})
b = b[b["n_ok"] >= 10][["name", "best_LD"]]
b = b.rename(columns={"best_LD": "LD_20k"})
both = a.merge(b, on="name")

print("Wings in both runs:", len(both))
link = both["LD_40k"].corr(both["LD_20k"])
print("Link between the two scores (-1 to +1):", round(link, 2))
print()
print("Top 10 wings at Re 20,000, with their score at Re 40,000:")
top = both.sort_values("LD_20k", ascending=False).head(10)
print(top[["name", "m", "p", "t", "LD_20k", "LD_40k"]].round(3).to_string(index=False))

base = both[both["name"] == "wing_000"].iloc[0]
print()
print("Baseline wing_000: Re 20,000 =", round(base["LD_20k"], 2),
      "  Re 40,000 =", round(base["LD_40k"], 2))
print("Best wing: Re 20,000 =", round(both["LD_20k"].max(), 2),
      "  Re 40,000 =", round(both["LD_40k"].max(), 2))

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(both["LD_40k"], both["LD_20k"], s=20)
ax[0].set_xlabel("score at Re 40,000")
ax[0].set_ylabel("score at Re 20,000")
for a_, col, label in [(ax[1], "t", "thickness (t)"),
                       (ax[2], "m", "curve amount (m)")]:
    a_.scatter(both[col], both["LD_20k"], s=20)
    a_.scatter([base[col]], [base["LD_20k"]], s=90, color="red",
               label="baseline")
    a_.set_xlabel(label)
    a_.set_ylabel("score at Re 20,000")
ax[1].legend()
for a_ in ax:
    a_.grid(True)
plt.tight_layout()
plt.savefig("graphs/compare_re_runs.png", dpi=150)
plt.show()