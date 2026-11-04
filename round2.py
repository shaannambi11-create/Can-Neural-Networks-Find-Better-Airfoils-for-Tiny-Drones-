import os
import subprocess
import time
import numpy as np
import pandas as pd

XFOIL = r"PATH\TO\xfoil.exe"
RE = 20000
N = 300                      # how many new thin wings
BEST_SO_FAR = 17.54          # best score from round 1 at Re 20,000

rng = np.random.default_rng(7)
folder = f"polars_re{RE}"
os.makedirs("data/wings", exist_ok=True)
os.makedirs(os.path.join("data", folder), exist_ok=True)
os.makedirs("results", exist_ok=True)

def naca4(m, p, t, n=80):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
              + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2),
                  m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    theta = np.arctan(dyc)
    xu = x - yt*np.sin(theta); yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta); yl = yc - yt*np.cos(theta)
    X = np.concatenate([xu[::-1], xl[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y

def run_xfoil(name):
    polar = f"{folder}/{name}.txt"
    full = os.path.join("data", polar)
    if os.path.exists(full):
        os.remove(full)
    cmds = ["plop", "g", "", f"load wings/{name}.dat", "", "pane",
            "oper", f"visc {RE}", "iter 200", "pacc", polar, "",
            "aseq -2 10 1", "pacc", "", "quit"]
    try:
        subprocess.run([XFOIL], input="\n".join(cmds) + "\n", text=True,
                       capture_output=True, cwd="data", timeout=120)
    except subprocess.TimeoutExpired:
        print("   (timed out)")
    return full

def read_polar(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            lines = f.readlines()
        idx = [i for i, l in enumerate(lines) if l.strip().startswith("---")]
        if not idx:
            return None
        cols = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
        df = pd.read_csv(path, skiprows=idx[0] + 1, sep=r"\s+", names=cols)
    except Exception:
        return None
    df = df[df["CD"] > 0].copy()
    if len(df) == 0:
        return None
    df["LD"] = df["CL"] / df["CD"]
    return df

rows = []
for i in range(N):
    m = round(rng.uniform(0.0, 0.08), 4)
    p = round(rng.uniform(0.2, 0.6), 4)
    t = round(rng.uniform(0.03, 0.08), 4)
    name = f"r2_{i:03d}"
    X, Y = naca4(m, p, t)
    with open(f"data/wings/{name}.dat", "w") as f:
        f.write(name + "\n")
        for xi, yi in zip(X, Y):
            f.write(f"{xi:.6f} {yi:.6f}\n")
    start = time.time()
    df = read_polar(run_xfoil(name))
    secs = round(time.time() - start, 1)
    if df is None:
        print(f"{name}: FAILED, no data ({secs}s)")
        rows.append({"name": name, "m": m, "p": p, "t": t,
                     "n_ok": 0, "best_LD": None, "best_alpha": None})
        continue
    best = df.loc[df["LD"].idxmax()]
    print(f"{name}: {len(df)}/13 angles ok, best L/D {best['LD']:.1f} at {best['alpha']:.0f} deg ({secs}s)")
    rows.append({"name": name, "m": m, "p": p, "t": t, "n_ok": len(df),
                 "best_LD": round(best["LD"], 2), "best_alpha": best["alpha"]})

res = pd.DataFrame(rows)
res.to_csv(f"results/results_round2_re{RE}.csv", index=False)
usable = res[res["n_ok"] >= 10]
print()
print(f"Round 2 at Re = {RE}")
print("Wings tested:", len(res))
print("Wings with 10 or more good angles:", len(usable))
print("Wings scoring above 15:", int((usable["best_LD"] > 15).sum()))
print()
print("Top 5 wings:")
print(usable.sort_values("best_LD", ascending=False).head(5).to_string(index=False))
print()
print("Best score in round 1:", BEST_SO_FAR)
print("Best score in round 2:", usable["best_LD"].max())