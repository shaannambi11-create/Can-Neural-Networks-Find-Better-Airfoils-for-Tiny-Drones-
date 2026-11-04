import pandas as pd
import matplotlib.pyplot as plt

FILE = "data/polar_re40000.txt"

with open(FILE) as f:
    lines = f.readlines()
start = next(i for i, l in enumerate(lines) if l.strip().startswith("---")) + 1

cols = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
df = pd.read_csv(FILE, skiprows=start, sep=r"\s+", names=cols)
df["LD"] = df["CL"] / df["CD"]

print(df[["alpha", "CL", "CD", "LD"]].round(3))
best = df.loc[df["LD"].idxmax()]
print("Best L/D:", round(best["LD"], 2), "at", best["alpha"], "deg")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(df["alpha"], df["CL"], "o-")
ax[0].set_xlabel("angle (deg)"); ax[0].set_ylabel("CL")
ax[1].plot(df["CD"], df["CL"], "o-")
ax[1].set_xlabel("CD"); ax[1].set_ylabel("CL")
ax[2].plot(df["alpha"], df["LD"], "o-")
ax[2].set_xlabel("angle (deg)"); ax[2].set_ylabel("L/D")
for a in ax:
    a.grid(True)
plt.tight_layout()
plt.savefig("graphs/polar_naca2412_re40000.png", dpi=150)
plt.show()