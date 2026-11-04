import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/results_re40000.csv")
good = df[df["n_ok"] >= 10].copy()
print("Wings used:", len(good), "of", len(df))
print("Left out:", list(df[df["n_ok"] < 10]["name"]))

base = good[good["name"] == "wing_000"]
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
names = [("t", "thickness (t)"), ("m", "curve amount (m)"), ("p", "curve position (p)")]
for a, (col, label) in zip(ax, names):
    a.scatter(good[col], good["best_LD"], s=25)
    a.scatter(base[col], base["best_LD"], s=90, color="red", label="baseline")
    a.set_xlabel(label)
    a.set_ylabel("best L/D")
    a.grid(True)
ax[0].legend()
plt.tight_layout()
plt.savefig("graphs/wings_vs_score.png", dpi=150)
plt.show()

print()
print("Link with score (-1 to +1):")
print(good[["t", "m", "p", "best_LD"]].corr()["best_LD"].round(2))