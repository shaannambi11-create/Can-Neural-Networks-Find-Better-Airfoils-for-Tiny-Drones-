import pandas as pd
import matplotlib.pyplot as plt

def load(file):
    with open(file) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("---")) + 1
    cols = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
    df = pd.read_csv(file, skiprows=start, sep=r"\s+", names=cols)
    df["LD"] = df["CL"] / df["CD"]
    return df

runs = [("Re 40,000", load("data/polar_re40000.txt")),
        ("Re 3,000,000", load("data/polar_re3m.txt"))]

for name, df in runs:
    best = df.loc[df["LD"].idxmax()]
    print(name, "-> best L/D:", round(best["LD"], 1), "at", best["alpha"], "deg")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for name, df in runs:
    ax[0].plot(df["alpha"], df["CL"], "o-", label=name)
    ax[1].plot(df["alpha"], df["CD"], "o-", label=name)
    ax[2].plot(df["alpha"], df["LD"], "o-", label=name)
for a, ylabel in zip(ax, ["CL (lift)", "CD (drag)", "L/D (score)"]):
    a.set_xlabel("angle (deg)")
    a.set_ylabel(ylabel)
    a.grid(True)
    a.legend()
plt.tight_layout()
plt.savefig("graphs/compare_re.png", dpi=150)
plt.show()