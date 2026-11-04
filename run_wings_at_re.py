import os
import subprocess
import time
import pandas as pd

XFOIL = r"PATH\TO\xfoil.exe"
RE = 20000        # change this number to test another Reynolds number
LIMIT = 500       # how many wings to test

params = pd.read_csv("data/wing_params.csv").head(LIMIT)
folder = f"polars_re{RE}"
os.makedirs(os.path.join("data", folder), exist_ok=True)
os.makedirs("results", exist_ok=True)

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
for _, w in params.iterrows():
    name = w["name"]
    start = time.time()
    df = read_polar(run_xfoil(name))
    secs = round(time.time() - start, 1)
    if df is None:
        print(f"{name}: FAILED, no data ({secs}s)")
        rows.append({**w.to_dict(), "n_ok": 0, "best_LD": None, "best_alpha": None})
        continue
    best = df.loc[df["LD"].idxmax()]
    print(f"{name}: {len(df)}/13 angles ok, best L/D {best['LD']:.1f} at {best['alpha']:.0f} deg ({secs}s)")
    rows.append({**w.to_dict(), "n_ok": len(df),
                 "best_LD": round(best["LD"], 2), "best_alpha": best["alpha"]})

res = pd.DataFrame(rows)
res.to_csv(f"results/results_re{RE}.csv", index=False)
usable = res[res["n_ok"] >= 10]
print()
print(f"Re = {RE}")
print("Wings tested:", len(res))
print("Wings with 10 or more good angles:", len(usable))
print()
print("Top 5 wings:")
print(usable.sort_values("best_LD", ascending=False).head(5).to_string(index=False))
base = res[res["name"] == "wing_000"].iloc[0]
print("Baseline wing_000 best L/D:", base["best_LD"])