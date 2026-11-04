import os
import sys
import glob
import subprocess
import time
from math import comb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ---------- how to run ----------
#   python scripts\repeat_search.py 1        (repeat number 1)
#   python scripts\repeat_search.py 2        (repeat number 2)
#   python scripts\repeat_search.py 3        (repeat number 3)
#   python scripts\repeat_search.py summary  (compare all repeats)

# ---------- settings ----------
XFOIL = r"PATH\TO\xfoil.exe"
RE = 20000
TEST = False           # True = tiny quick test
if TEST:
    N0, ROUNDS, K = 20, 2, 8
else:
    N0, ROUNDS, K = 400, 6, 40     # start wings, AI rounds, wings per round
TIMEOUT = 40           # seconds XFOIL may spend on one wing

# filter settings (same as your ai_loop_filtered.py)
CD_FLOOR = 0.010
CDP_MIN = -0.0005
MAX_CL_SLOPE = 0.30
TEACH_AI_REJECTS = True

# ---------- wing shape: 8 numbers ----------
UP = (0.02, 0.35)
LO = (-0.30, 0.15)
LOW = np.array([UP[0]] * 4 + [LO[0]] * 4)
HIGH = np.array([UP[1]] * 4 + [LO[1]] * 4)
XS = np.linspace(0, 1, 201)
WCOLS = [f"w{i}" for i in range(8)]

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

# ---------- XFOIL + filter ----------
folder = f"polars_re{RE}"
os.makedirs("data/wings", exist_ok=True)
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
                       capture_output=True, cwd="data", timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        pass
    return full

def check_polar(d):
    if (d["CDp"] < CDP_MIN).any():
        return "negative pressure drag"
    if (d["CD"] < CD_FLOOR).any():
        return "drag too low"
    d2 = d.sort_values("alpha")
    slope = d2["CL"].diff() / d2["alpha"].diff()
    if (slope.abs() > MAX_CL_SLOPE).any():
        return "lift jumps"
    return ""

def read_polar(path):
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

def usable(df):
    return df[(df["n_ok"] >= 10) & df["best_LD"].notna()]

def training_set(df):
    d = usable(df)
    if not TEACH_AI_REJECTS:
        return d
    bad = df[df["reject"].fillna("") != ""].copy()
    bad["best_LD"] = 0.0
    return pd.concat([d, bad])

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
    if len(d) == 0:
        print(f"   {label}: no good wings yet")
        return
    b = d.loc[d["best_LD"].idxmax()]
    print(f"   {label}: best score so far = {b['best_LD']:.2f} ({b['name']})")

# ---------- compare repeats ----------
def compare(df):
    good = usable(df)
    ai = good[good["source"].isin(["random", "ai"])].sort_values("order")
    rd = good[good["source"].isin(["random", "random_arm"])].sort_values("order")
    n = min(len(ai), len(rd))
    return n, ai["best_LD"].head(n).max(), rd["best_LD"].head(n).max()

def summary():
    files = sorted(glob.glob("results/rep[0-9]*.csv"))
    files = [f for f in files if "_test" not in f]
    if not files:
        print("No repeat files found in results.")
        return
    print("Repeat   good wings   passed(random/AI/random-pick)   best AI   best random")
    wins = 0
    for f in files:
        df = pd.read_csv(f)
        if not (df["source"] == "random_arm").any():
            print(f"{os.path.basename(f)}: not finished yet, skipping")
            continue
        n, a, r = compare(df)
        rates = []
        for src in ["random", "ai", "random_arm"]:
            t = df[df["source"] == src]
            p = usable(t)
            rates.append(f"{100 * len(p) / max(len(t), 1):.0f}%")
        print(f"{os.path.basename(f):12s} {n:6d}       {'/'.join(rates):>22s}      {a:7.2f}   {r:7.2f}")
        wins += int(a > r)
    print()
    print(f"AI-guided search won in {wins} of {len(files)} repeats.")
    print("NACA 2412 baseline: 3.95")

# ---------- one repeat ----------
def evaluate(W, source, rnd, start_index, df, seed, data_file):
    rows = []
    for k, w in enumerate(W):
        idx = start_index + k
        name = f"s{seed}_c{idx:05d}"
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
        if (k + 1) % 10 == 0 or k + 1 == len(W):
            print(f"   ...{k + 1} of {len(W)} wings done", flush=True)
    new = pd.DataFrame(rows)
    df = pd.concat([df, new], ignore_index=True) if df is not None else new
    df.to_csv(data_file, index=False)
    return df

def run_repeat(seed):
    data_file = f"results/rep{seed}_test.csv" if TEST else f"results/rep{seed}.csv"
    rng = np.random.default_rng(100 + seed)
    t0 = time.time()
    total = N0 + 2 * ROUNDS * K
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        print(f"Repeat {seed}: found earlier data, continuing ({len(df)} wings)")
    else:
        print(f"Repeat {seed}, round 0: testing {N0} random wings...")
        df = evaluate(random_wings(rng, N0), "random", 0, 0, None, seed, data_file)
        best_line(df, "Round 0")
        mins = (time.time() - t0) / 60
        print(f"   Round 0 took {mins:.1f} minutes.")
        print(f"   This whole repeat should take about {mins * total / N0:.0f} minutes.")
    done = int(df[df["source"] == "ai"]["round"].max()) if (df["source"] == "ai").any() else 0
    for r in range(done + 1, ROUNDS + 1):
        print(f"Repeat {seed}, round {r}: AI is learning and searching...")
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
        df = evaluate(np.array(chosen), "ai", r, len(df), df, seed, data_file)
        best_line(df, f"After round {r}")
    if not (df["source"] == "random_arm").any():
        print(f"Repeat {seed}, fair test: {ROUNDS * K} random wings...")
        df = evaluate(random_wings(rng, ROUNDS * K), "random_arm", 0, len(df),
                      df, seed, data_file)
    n, a, r_ = compare(df)
    print()
    print(f"Repeat {seed} finished in {(time.time() - t0) / 60:.1f} minutes.")
    print(f"   Equal number of good wings: {n}")
    print(f"   Best score, AI-guided: {a}")
    print(f"   Best score, random   : {r_}")

if len(sys.argv) < 2:
    print("Please add a repeat number, like:  python scripts\\repeat_search.py 1")
elif sys.argv[1] == "summary":
    summary()
else:
    run_repeat(int(sys.argv[1]))
