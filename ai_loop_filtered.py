import os
import sys
import subprocess
import time
from math import comb
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ---------- settings ----------
XFOIL = r"PATH\TO\xfoil.exe"
RE = 20000
TEST = False
if TEST:
    N0, ROUNDS, K = 30, 2, 10        # start wings, AI rounds, wings per round
else:
    N0, ROUNDS, K = 800, 8, 60
BASELINE = 3.95        # NACA 2412 score at Re 20,000 from earlier
DATA_FILE = "results/cst_data_test.csv" if TEST else "results/cst_data.csv"

# ---------- NEW: filter settings (a wing with ANY broken row is rejected) ----------
CD_FLOOR = 0.010       # drag below this is too small to be real at Re 20,000
CDP_MIN = -0.0005      # pressure drag below this is negative = broken
MAX_CL_SLOPE = 0.30    # lift can't change more than this per 1 degree
REDO_AI_ROUNDS = False # True = throw away old AI rounds and redo them all with the filter
TEACH_AI_REJECTS = True  # True = AI learns that rejected wings are bad (score 0)

# ---------- wing shape: 8 numbers (4 for the top, 4 for the bottom) ----------
UP = (0.02, 0.35)      # allowed range for the 4 top numbers
LO = (-0.30, 0.15)     # allowed range for the 4 bottom numbers
LOW = np.array([UP[0]] * 4 + [LO[0]] * 4)
HIGH = np.array([UP[1]] * 4 + [LO[1]] * 4)
XS = np.linspace(0, 1, 201)

def surface(x, w):
    n = len(w) - 1
    K_ = sum(w[i] * comb(n, i) * x**i * (1 - x)**(n - i) for i in range(n + 1))
    return np.sqrt(x) * (1 - x) * K_

def stats(w):
    yu = surface(XS, w[:4])
    yl = surface(XS, w[4:])
    th = yu - yl
    cam = (yu + yl) / 2
    return th.min(), th.max(), cam.max(), XS[np.argmax(cam)]

def valid(w):
    tmin, tmax, cmax, _ = stats(w)
    return tmin >= -1e-6 and 0.02 <= tmax <= 0.10 and 0.0 <= cmax <= 0.09

def features(W):
    rows = []
    for w in W:
        _, tmax, cmax, cpos = stats(w)
        rows.append(list(w) + [tmax, cmax, cpos])
    return np.array(rows)

def random_wings(rng, n):
    out = []
    while len(out) < n:
        w = rng.uniform(LOW, HIGH)
        if valid(w):
            out.append(w)
    return np.array(out)

def mutants(rng, parents, n):
    out = []
    while len(out) < n:
        p = parents[rng.integers(len(parents))]
        w = np.clip(p + rng.normal(0, 0.04, 8) * (HIGH - LOW), LOW, HIGH)
        if valid(w):
            out.append(w)
    return np.array(out)

def write_dat(name, w, n=80):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yu = surface(x, w[:4])
    yl = surface(x, w[4:])
    X = np.concatenate([x[::-1], x[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    with open(f"data/wings/{name}.dat", "w") as f:
        f.write(name + "\n")
        for xi, yi in zip(X, Y):
            f.write(f"{xi:.6f} {yi:.6f}\n")

# ---------- XFOIL ----------
folder = f"polars_re{RE}"
os.makedirs("data/wings", exist_ok=True)
os.makedirs(os.path.join("data", folder), exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

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
                       capture_output=True, cwd="data", timeout=40)
    except subprocess.TimeoutExpired:
        pass
    return full

# ---------- NEW: the filter ----------
def check_polar(d):
    """Look at every row of a polar. Return a reason text if the wing is broken,
    or "" if the polar looks fine."""
    if (d["CDp"] < CDP_MIN).any():
        return "negative pressure drag"
    if (d["CD"] < CD_FLOOR).any():
        return "drag too low"
    d2 = d.sort_values("alpha")
    slope = d2["CL"].diff() / d2["alpha"].diff()     # lift change per degree
    if (slope.abs() > MAX_CL_SLOPE).any():
        return "lift jumps"
    return ""

def read_polar(path):
    """Returns (table, reason).
    table = the good polar, or None if no polar / wing rejected.
    reason = "" normally, or the filter reason if the wing was rejected."""
    if not os.path.exists(path):
        return None, ""
    try:
        with open(path) as f:
            lines = f.readlines()
        idx = [i for i, l in enumerate(lines) if l.strip().startswith("---")]
        if not idx:
            return None, ""
        cols = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
        df = pd.read_csv(path, skiprows=idx[0] + 1, sep=r"\s+", names=cols)
    except Exception:
        return None, ""
    if len(df) == 0:
        return None, ""
    why = check_polar(df)
    if why:
        return None, why
    df = df[df["CD"] > 0].copy()
    if len(df) == 0:
        return None, ""
    df["LD"] = df["CL"] / df["CD"]
    return df, ""

# ---------- NEW: check one polar file without running the loop ----------
# Use:  python scripts\ai_loop.py check data\polars_re20000\c01025.txt
if len(sys.argv) >= 3 and sys.argv[1] == "check":
    res, why = read_polar(sys.argv[2])
    if why:
        print("REJECTED:", why)
    elif res is None:
        print("No good polar found in this file.")
    else:
        print("PASS. Best score =", round(res["LD"].max(), 2),
              "at alpha", res.loc[res["LD"].idxmax(), "alpha"])
    sys.exit()

def evaluate(W, source, rnd, start_index, df):
    """Run XFOIL on each wing in W and add the results to the table."""
    rows = []
    for k, w in enumerate(W):
        idx = start_index + k
        name = f"c{idx:05d}"
        write_dat(name, w)
        res, why = read_polar(run_xfoil(name))
        _, tmax, cmax, cpos = stats(w)
        row = {"name": name, "order": idx, "round": rnd, "source": source}
        for i in range(8):
            row[f"w{i}"] = w[i]
        row.update({"t": tmax, "m": cmax, "p": cpos, "reject": why})
        if res is None:
            row.update({"n_ok": 0, "best_LD": np.nan, "best_alpha": np.nan})
        else:
            b = res.loc[res["LD"].idxmax()]
            row.update({"n_ok": len(res), "best_LD": round(b["LD"], 2),
                        "best_alpha": b["alpha"]})
        rows.append(row)
    new = pd.DataFrame(rows)
    df = pd.concat([df, new], ignore_index=True) if df is not None else new
    df.to_csv(DATA_FILE, index=False)
    return df

WCOLS = [f"w{i}" for i in range(8)]

def usable(df):
    return df[(df["n_ok"] >= 10) & df["best_LD"].notna()]

def training_set(df):
    """Good wings with their score. Optionally add rejected wings with score 0,
    so the AI learns to stay away from them."""
    d = usable(df)
    if not TEACH_AI_REJECTS:
        return d
    bad = df[df["reject"].fillna("") != ""].copy()
    bad["best_LD"] = 0.0
    return pd.concat([d, bad])

# ---------- NEW: re-score old saved wings with the filter ----------
def rescore(df):
    backup = DATA_FILE.replace(".csv", "_before_filter.csv")
    if not os.path.exists(backup):
        df.to_csv(backup, index=False)
        print("   Saved a copy of your old results:", backup)
    old_best = usable(df)["best_LD"].max()
    df["reject"] = ""
    df["reject"] = df["reject"].astype(object)
    for i in df.index:
        path = os.path.join("data", folder, f"{df.at[i, 'name']}.txt")
        if not os.path.exists(path):
            continue                      # polar file missing: keep old numbers
        res, why = read_polar(path)
        df.at[i, "reject"] = why
        if res is None:
            df.at[i, "n_ok"] = 0
            df.at[i, "best_LD"] = np.nan
            df.at[i, "best_alpha"] = np.nan
        else:
            b = res.loc[res["LD"].idxmax()]
            df.at[i, "n_ok"] = len(res)
            df.at[i, "best_LD"] = round(b["LD"], 2)
            df.at[i, "best_alpha"] = b["alpha"]
    print("   Filter result on old wings:")
    counts = df["reject"].value_counts()
    for why, n in counts.items():
        if why:
            print(f"      rejected, {why}: {n}")
    print(f"      total rejected: {int((df['reject'] != '').sum())} of {len(df)}")
    print(f"      best score before filter: {old_best}")
    print(f"      best score after filter : {usable(df)['best_LD'].max()}")
    df.to_csv(DATA_FILE, index=False)
    return df

def train(df):
    d = training_set(df)
    X = features(d[WCOLS].values)
    y = d["best_LD"].values
    models = []
    for s in range(5):
        mdl = make_pipeline(StandardScaler(),
                            MLPRegressor(hidden_layer_sizes=(64, 64),
                                         max_iter=3000, random_state=s))
        models.append(mdl.fit(X, y))
    return models

def predict(models, W):
    X = features(W)
    P = np.array([m.predict(X) for m in models])
    return P.mean(axis=0), P.std(axis=0)

def holdout_report(df):
    d = usable(df)
    X = features(d[WCOLS].values)
    y = d["best_LD"].values
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=0)
    mdl = make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(64, 64),
                                     max_iter=3000, random_state=0)).fit(Xa, ya)
    g = mdl.predict(Xb)
    print(f"   AI check on unseen wings: R2 = {r2_score(yb, g):.2f}, "
          f"average miss = {mean_absolute_error(yb, g):.2f}")

def best_line(df, label):
    d = usable(df)
    b = d.loc[d["best_LD"].idxmax()]
    print(f"   {label}: best score so far = {b['best_LD']:.2f} ({b['name']})")

# ---------- main ----------
rng = np.random.default_rng(11)
t0 = time.time()
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    print("Found earlier data, continuing:", len(df), "wings")
    print("Re-checking all old wings with the new filter...")
    df = rescore(df)
    if REDO_AI_ROUNDS:
        df = df[df["source"] == "random"].reset_index(drop=True)
        df.to_csv(DATA_FILE, index=False)
        print("   Old AI rounds removed. They will be redone with the filter.")
    best_line(df, "After filter")
else:
    print(f"Round 0: testing {N0} random wings in XFOIL...")
    df = evaluate(random_wings(rng, N0), "random", 0, 0, None)
    best_line(df, "Round 0")

done = int(df[df["source"] == "ai"]["round"].max()) if (df["source"] == "ai").any() else 0
for r in range(done + 1, ROUNDS + 1):
    print(f"Round {r}: AI is learning and searching...")
    if r == done + 1:
        holdout_report(df)
    models = train(df)
    d = usable(df)
    top = d.sort_values("best_LD", ascending=False).head(100)[WCOLS].values
    cands = np.vstack([random_wings(rng, 20000), mutants(rng, top, 20000)])
    mean, std = predict(models, cands)
    order = np.argsort(mean + std)[::-1]
    scale = HIGH - LOW
    have = [w for w in df[WCOLS].values]
    chosen = []
    for i in order:
        w = cands[i]
        if all(np.max(np.abs(w - h) / scale) > 0.04 for h in have + chosen):
            chosen.append(w)
        if len(chosen) == K:
            break
    df = evaluate(np.array(chosen), "ai", r, len(df), df)
    best_line(df, f"After round {r}")

if not (df["source"] == "random_arm").any():
    n_rand = ROUNDS * K
    print(f"Fair test: {n_rand} more random wings in XFOIL...")
    df = evaluate(random_wings(rng, n_rand), "random_arm", 0, len(df), df)

# ---------- results ----------
d = usable(df)
ai = d[d["source"].isin(["random", "ai"])].sort_values("order")
rd = d[d["source"].isin(["random", "random_arm"])].sort_values("order")
best_ai = ai["best_LD"].max()
best_rd = rd["best_LD"].max()
print()
print("Number of XFOIL runs for each side:", len(ai), "(AI-guided) and", len(rd), "(random only)")
print("Best score, AI-guided search:", best_ai)
print("Best score, random search   :", best_rd)
print("NACA 2412 baseline          :", BASELINE)
print()
print("Top 5 wings found:")
cols = ["name", "source", "t", "m", "p", "n_ok", "best_LD", "best_alpha"]
print(d.sort_values("best_LD", ascending=False).head(5)[cols].round(4).to_string(index=False))

top1 = d.loc[d["best_LD"].idxmax()]
with open(f"data/wings/{top1['name']}.dat") as src, open("results/best_wing.dat", "w") as dst:
    dst.write(src.read())

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for dd, label in [(ai, "AI-guided"), (rd, "random")]:
    ax[0].plot(range(1, len(dd) + 1), dd["best_LD"].cummax().values, label=label)
ax[0].axhline(BASELINE, color="gray", linestyle="--", label="NACA 2412")
ax[0].set_xlabel("wings tested (successful XFOIL runs)")
ax[0].set_ylabel("best score so far")
ax[0].legend()
ax[0].grid(True)
for _, row in d.sort_values("best_LD", ascending=False).head(3).iterrows():
    w = row[WCOLS].values.astype(float)
    ax[1].plot(XS, surface(XS, w[:4]), label=row["name"])
    ax[1].plot(XS, surface(XS, w[4:]), color=ax[1].lines[-1].get_color())
ax[1].set_aspect("equal")
ax[1].set_title("Top 3 wing shapes")
ax[1].legend()
plt.tight_layout()
plt.savefig("graphs/ai_vs_random_search.png", dpi=150)
print()
print(f"Done in {round((time.time() - t0) / 60, 1)} minutes. Graph saved.")
plt.show()