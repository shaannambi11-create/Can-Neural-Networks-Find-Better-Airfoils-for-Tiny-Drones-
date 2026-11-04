import os
import re
import sys
import glob
import shutil
import subprocess
from math import comb
import numpy as np
import pandas as pd

# =====================================================================
#  How to run (in the black window, inside the aero-project folder):
#    python scripts\extra_analysis.py validation   (numbers for the X and Y gaps)
#    python scripts\extra_analysis.py summary      (AI vs random over all repeats)
#    python scripts\extra_analysis.py filters      (do results depend on the filter?)
#    python scripts\extra_analysis.py reynolds     (best wings at other Reynolds numbers)
#    python scripts\extra_analysis.py share        (make a folder to share code and data)
# =====================================================================

XFOIL = r"PATH\TO\xfoil.exe"
RE_MAIN = 20000
POLAR_DIR = f"data/polars_re{RE_MAIN}"
MAIN = {"cd_floor": 0.010, "slope": 0.30, "cdp_min": -0.0005}   # your filter settings

# published wind-tunnel points: NACA 6409, Re = 61,400 (Selig et al. 1995, Vol. 1)
MEASURED = [(-2.80, -0.005, 0.0628), (-1.14, 0.240, 0.0386), (0.26, 0.454, 0.0383),
            (2.07, 0.649, 0.0391), (3.32, 0.770, 0.0430), (4.98, 0.911, 0.0509)]

WCOLS = [f"w{i}" for i in range(8)]
UP, LO = (0.02, 0.35), (-0.30, 0.15)
LOW = np.array([UP[0]] * 4 + [LO[0]] * 4)
HIGH = np.array([UP[1]] * 4 + [LO[1]] * 4)

# ---------------------------------------------------------------- shapes
def surface(x, w):
    n = len(w) - 1
    K_ = sum(w[i] * comb(n, i) * x**i * (1 - x)**(n - i) for i in range(n + 1))
    return np.sqrt(x) * (1 - x) * K_

def naca4_points(m, p, t, n=80):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu = x - yt*np.sin(th); yu = yc + yt*np.cos(th)
    xl = x + yt*np.sin(th); yl = yc - yt*np.cos(th)
    return np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])

def cst_points(w, n=80):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yu, yl = surface(x, w[:4]), surface(x, w[4:])
    return np.concatenate([x[::-1], x[1:]]), np.concatenate([yu[::-1], yl[1:]])

def write_dat(name, X, Y):
    with open(f"data/wings/{name}.dat", "w") as f:
        f.write(name + "\n")
        for xi, yi in zip(X, Y):
            f.write(f"{xi:.6f} {yi:.6f}\n")

# ---------------------------------------------------------------- polars + filter
def read_raw(path):
    """Read an XFOIL polar file without any filter. Returns a table or None."""
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
    return df if len(df) else None

def judge(df, cd_floor, slope, cdp_min):
    """Return (n_ok, best_LD, best_alpha) or (0, nan, nan) if the wing fails."""
    if df is None:
        return 0, np.nan, np.nan
    if (df["CDp"] < cdp_min).any() or (df["CD"] < cd_floor).any():
        return 0, np.nan, np.nan
    d2 = df.sort_values("alpha")
    sl = d2["CL"].diff() / d2["alpha"].diff()
    if (sl.abs() > slope).any():
        return 0, np.nan, np.nan
    d = df[df["CD"] > 0].copy()
    if len(d) < 10:
        return 0, np.nan, np.nan
    d["LD"] = d["CL"] / d["CD"]
    b = d.loc[d["LD"].idxmax()]
    return len(d), float(b["LD"]), float(b["alpha"])

def load_repeats():
    found = {}
    for f in glob.glob("results/rep[0-9]*.csv"):
        if "_test" in f:
            continue
        num = int(re.findall(r"rep(\d+)", os.path.basename(f))[0])
        df = pd.read_csv(f)
        if (df["source"] == "random_arm").any():           # finished repeats only
            found[num] = df
    return {k: found[k] for k in sorted(found)}

def compare(df, ok_col="n_ok", best_col="best_LD"):
    good = df[(df[ok_col] >= 10) & df[best_col].notna()]
    ai = good[good["source"].isin(["random", "ai"])].sort_values("order")
    rd = good[good["source"].isin(["random", "random_arm"])].sort_values("order")
    n = min(len(ai), len(rd))
    if n == 0:
        return 0, np.nan, np.nan
    return n, ai[best_col].head(n).max(), rd[best_col].head(n).max()

def sign_summary(diffs):
    diffs = np.array(diffs, dtype=float)
    wins, ties, losses = int((diffs > 0).sum()), int((diffs == 0).sum()), int((diffs < 0).sum())
    return wins, ties, losses

# ================================================================ validation
def cmd_validation():
    v = pd.read_csv("results/xfoil_validation.csv")
    v = v[v["airfoil"] == "NACA 6409"]
    print("XFOIL compared with the wind tunnel (NACA 6409, Re = 61,400)")
    print("Angles used: the measured angles inside XFOIL's range, up to 6 degrees")
    print()
    print("  setting n    average lift miss    average drag miss")
    out = {}
    for n in sorted(v["n"].unique()):
        d = v[(v["n"] == n) & (v["alpha"] <= 6)].sort_values("alpha")
        lm, dm = [], []
        for a, cl_m, cd_m in MEASURED:
            if d["alpha"].min() <= a <= d["alpha"].max():
                lm.append(abs(np.interp(a, d["alpha"], d["CL"]) - cl_m))
                dm.append(abs(100 * (np.interp(a, d["alpha"], d["CD"]) - cd_m) / cd_m))
        out[n] = (np.mean(lm), np.mean(dm), len(lm))
        print(f"  {n:6d}      {np.mean(lm):10.3f}           {np.mean(dm):8.0f}%      ({len(lm)} angles)")
    print()
    if 5 in out and 12 in out:
        print("Sentence for the paper:")
        print(f"  Ncrit = 5 and 12 gave average lift differences of {out[5][0]:.3f} and {out[12][0]:.3f} "
              f"and drag differences of {out[5][1]:.0f}% and {out[12][1]:.0f}%.")

# ================================================================ summary
def cmd_summary():
    reps = load_repeats()
    if not reps:
        print("No finished repeats found in results (rep1.csv, rep2.csv, ...).")
        return
    print("Repeat   good wings each side   best AI   best random   difference")
    diffs = []
    for k, df in reps.items():
        n, a, r = compare(df)
        diffs.append(a - r)
        print(f"{k:6d}   {n:12d}           {a:7.2f}   {r:9.2f}   {a - r:+9.2f}")
    wins, ties, losses = sign_summary(diffs)
    print()
    print(f"Repeats finished: {len(diffs)}")
    print(f"AI better: {wins}   tied: {ties}   AI worse: {losses}")
    print(f"Average difference (AI minus random): {np.mean(diffs):+.2f}   middle value: {np.median(diffs):+.2f}")
    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(diffs, len(diffs))) for _ in range(10000)]
    print(f"Likely range of the average difference (95%): {np.percentile(boots, 2.5):+.2f} to {np.percentile(boots, 97.5):+.2f}")
    try:
        from scipy import stats
        if wins + losses > 0:
            p = stats.binomtest(wins, wins + losses, 0.5).pvalue
            print(f"Sign test p-value (ties left out): {p:.3f}")
        if any(d != 0 for d in diffs):
            print(f"Wilcoxon test p-value: {stats.wilcoxon(diffs).pvalue:.3f}")
    except Exception as e:
        print("(could not run the tests:", e, ")")
    print()
    print("How to read this:")
    print(" - A p-value is the chance of seeing a result this lopsided if AI and random were")
    print("   really equally good. Small (below 0.05) points to a real difference.")
    print(" - It is not proof. Report the wins, ties, losses, and the range, honestly.")

# ================================================================ filters
def cmd_filters():
    reps = load_repeats()
    if not reps:
        print("No finished repeats found.")
        return
    cache = {}
    for df in reps.values():
        for name in df["name"]:
            if name not in cache:
                cache[name] = read_raw(os.path.join(POLAR_DIR, f"{name}.txt"))
    have = sum(v is not None for v in cache.values())
    print(f"Polar files found: {have} of {len(cache)} wings in the repeats")
    if have == 0:
        print("No polar files found in", POLAR_DIR)
        return
    settings = [("no filter", dict(cd_floor=0.0, slope=1e9, cdp_min=-1e9))]
    for cdf in (0.005, 0.010, 0.015):
        for sl in (0.20, 0.30, 0.40):
            settings.append((f"CD floor {cdf:.3f}, lift slope {sl:.2f}",
                             dict(cd_floor=cdf, slope=sl, cdp_min=MAIN["cdp_min"])))
    for cp in (-0.002, 0.0):
        settings.append((f"negative-drag limit {cp}", dict(cd_floor=MAIN["cd_floor"], slope=MAIN["slope"], cdp_min=cp)))
    base_ok = sum(1 for v in cache.values() if judge(v, 0.0, 1e9, -1e9)[0] >= 10)
    print()
    print(f"{'setting':42s} {'rejected':>9s} {'AI better':>10s} {'tied':>5s} {'AI worse':>9s} {'mean AI - random':>17s}")
    for label, s in settings:
        res = {n: judge(v, **s) for n, v in cache.items()}
        passed = sum(1 for r in res.values() if r[0] >= 10)
        rejected = 100 * (1 - passed / base_ok) if base_ok else float("nan")
        diffs = []
        for df in reps.values():
            d = df.copy()
            d["n_ok"] = [res[nm][0] for nm in d["name"]]
            d["best_LD"] = [res[nm][1] for nm in d["name"]]
            n, a, r = compare(d)
            diffs.append(a - r)
        w, t, l = sign_summary(diffs)
        mark = "  <- your settings" if s == dict(MAIN) else ""
        print(f"{label:42s} {rejected:8.0f}% {w:10d} {t:5d} {l:9d} {np.mean(diffs):+17.2f}{mark}")
    print()
    print("Note: the filter changes which wings count as good. It does not change which wings")
    print("the AI chose, because those choices were made with your settings. To test that too,")
    print("the whole search would have to be run again with each setting.")

# ================================================================ reynolds
def run_xfoil_for(file_name, re_num, out_name):
    polar = f"polars_check/{out_name}.txt"
    full = os.path.join("data", polar)
    if os.path.exists(full):
        os.remove(full)
    cmds = ["plop", "g", "", f"load wings/{file_name}.dat", "", "pane",
            "oper", f"visc {re_num}", "iter 200", "pacc", polar, "",
            "aseq -2 10 1", "pacc", "", "quit"]
    try:
        subprocess.run([XFOIL], input="\n".join(cmds) + "\n", text=True,
                       capture_output=True, cwd="data", timeout=60)
    except subprocess.TimeoutExpired:
        pass
    return full

def cmd_reynolds():
    import matplotlib
    import matplotlib.pyplot as plt
    os.makedirs("data/polars_check", exist_ok=True)
    os.makedirs("graphs/paper", exist_ok=True)
    frames = []
    for f in sorted(glob.glob("results/rep[0-9]*.csv")) + ["results/cst_data.csv"]:
        if os.path.exists(f) and "_test" not in f and "before_filter" not in f:
            frames.append(pd.read_csv(f))
    allw = pd.concat(frames, ignore_index=True)
    allw = allw[(allw["n_ok"] >= 10) & allw["best_LD"].notna()].sort_values("best_LD", ascending=False)
    scale = HIGH - LOW
    chosen = []
    for _, row in allw.iterrows():
        w = row[WCOLS].values.astype(float)
        if all(np.max(np.abs(w - c[WCOLS].values.astype(float)) / scale) > 0.10 for c in chosen):
            chosen.append(row)
        if len(chosen) == 5:
            break
    wings = []
    for k, row in enumerate(chosen, 1):
        X, Y = cst_points(row[WCOLS].values.astype(float))
        wings.append((f"AI wing {k}", f"chk_ai{k}", X, Y, "ai"))
    for label, (m, p, t) in [("NACA 2412 (12% thick)", (0.02, 0.4, 0.12)),
                             ("NACA 6409 (9% thick)", (0.06, 0.4, 0.09)),
                             ("NACA 5403 (3% thick)", (0.05, 0.4, 0.03))]:
        X, Y = naca4_points(m, p, t)
        wings.append((label, "chk_" + label.split()[1], X, Y, "std"))
    for label, fname, X, Y, kind in wings:
        write_dat(fname, X, Y)
    res_list = [10000, 20000, 30000, 40000, 60000, 100000]
    table = {}
    print(f"Testing {len(wings)} wings at {len(res_list)} Reynolds numbers ({len(wings) * len(res_list)} XFOIL runs)...")
    for label, fname, X, Y, kind in wings:
        for re_num in res_list:
            path = run_xfoil_for(fname, re_num, f"{fname}_re{re_num}")
            n_ok, best, ang = judge(read_raw(path), **MAIN)
            table[(label, re_num)] = best
        print("  done:", label, flush=True)
    rows = []
    print()
    print(f"{'wing':24s}" + "".join(f"{r:>9d}" for r in res_list))
    for label, fname, X, Y, kind in wings:
        vals = [table[(label, r)] for r in res_list]
        rows.append({"wing": label, **{f"Re_{r}": v for r, v in zip(res_list, vals)}})
        print(f"{label:24s}" + "".join(f"{v:9.1f}" if not np.isnan(v) else "     fail" for v in vals))
    pd.DataFrame(rows).to_csv("results/reynolds_check.csv", index=False)
    print()
    print("A score of 'fail' means XFOIL did not give a result that passed the checks.")

    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"]})
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for label, fname, X, Y, kind in wings:
        ys = [table[(label, r)] for r in res_list]
        if kind == "ai":
            ax.plot(res_list, ys, "o-", color="#1f5fa8", alpha=0.7, ms=3, lw=0.9,
                    label="AI-found wings" if label == "AI wing 1" else None)
        else:
            col = {"NACA 2412": "#c0392b", "NACA 6409": "#e08e0b", "NACA 5403": "#6a3d9a"}[label.split(" (")[0]]
            ax.plot(res_list, ys, "s--", color=col, ms=3, lw=0.9, label=label)
    ax.set_xscale("log")
    ax.set_xticks(res_list)
    ax.set_xticklabels(["10k", "20k", "30k", "40k", "60k", "100k"])
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel("Score (lift-to-drag ratio)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig("graphs/paper/nambi_SI_reynolds.jpg", dpi=600, pil_kwargs={"quality": 95})
    print("Graph saved: graphs\\paper\\nambi_SI_reynolds.jpg")

# ================================================================ share
README = """# AI-guided wing design for tiny, slow drones (code and results)

This folder lets other people repeat the study. It goes with the paper
"AI-Guided Wing Design for Tiny, Slow Drones: A Computer Study at Low Reynolds Number".

## What is inside
- scripts/   the Python programs I used
- results/   tables of results (one row per wing tested)
- SETTINGS.txt   the settings used
- requirements.txt   the Python libraries and their versions

## What you need
- Python 3 and the libraries in requirements.txt (pip install -r requirements.txt)
- XFOIL 6.99 (free, from MIT: web.mit.edu/drela/Public/web/xfoil). It is not included here.
  Open each script and set XFOIL = r"PATH\\TO\\xfoil.exe" at the top.

## How to repeat the main experiment
1. Make a folder with the sub-folders data, results, scripts and graphs.
2. Put the scripts in scripts.
3. Run: python scripts\\repeat_search.py 1   (then 2, 3, ...)
4. Compare all repeats: python scripts\\extra_analysis.py summary
Each repeat can take 1.5 to 3 hours.

## Important limits
- All results come from computer models (XFOIL, two dimensions, Mach 0). No real wings were built.
- My quality filter settings are my own judgement (see SETTINGS.txt).

## License
Code: MIT License (see LICENSE). Results tables: free to reuse with credit.
"""

MIT = """MIT License

Copyright (c) 2026 Shaan Gopal Nambi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def cmd_share():
    from importlib import metadata
    out = "share"
    for sub in ("scripts", "results"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    n_scripts = 0
    for f in glob.glob("scripts/*.py"):
        text = open(f, encoding="utf8", errors="ignore").read()
        text = re.sub(r'^XFOIL\s*=\s*r?".*"', 'XFOIL = r"PATH\\\\TO\\\\xfoil.exe"', text, flags=re.M)
        with open(os.path.join(out, "scripts", os.path.basename(f)), "w", encoding="utf8") as g:
            g.write(text)
        n_scripts += 1
    n_results = 0
    for f in glob.glob("results/*.csv"):
        if "_test" in f or "before_filter" in f:
            continue
        shutil.copy(f, os.path.join(out, "results", os.path.basename(f)))
        n_results += 1
    for f in ("results/best_wing.dat", "data/wing_params.csv"):
        if os.path.exists(f):
            shutil.copy(f, os.path.join(out, "results", os.path.basename(f)))
    libs = ["numpy", "pandas", "scikit-learn", "matplotlib", "scipy", "joblib"]
    with open(os.path.join(out, "requirements.txt"), "w") as f:
        for lib in libs:
            try:
                f.write(f"{lib}=={metadata.version(lib)}\n")
            except Exception:
                f.write(f"{lib}\n")
    with open(os.path.join(out, "README.md"), "w", encoding="utf8") as f:
        f.write(README)
    with open(os.path.join(out, "LICENSE"), "w") as f:
        f.write(MIT)
    with open(os.path.join(out, "SETTINGS.txt"), "w") as f:
        f.write(
"""Settings used in the study
--------------------------
Analysis program: XFOIL 6.99, viscous mode, Mach 0, Ncrit = 9, automatic repaneling, 200 iterations
Main Reynolds number: 20,000 (also 40,000 for early NACA four-digit runs)
Angles: -2 to 10 degrees in steps of 1 (13 angles). Score = best CL/CD.
A wing counts only if XFOIL worked at 10 or more of the 13 angles and passed the filter.
Quality filter (my own judgement): reject if any angle has pressure drag below -0.0005,
total drag below 0.010, or lift change above 0.30 per degree.
Wing shapes: CST method, 4 numbers for the top and 4 for the bottom
  top numbers 0.02 to 0.35, bottom numbers -0.30 to 0.15
  thickness 2% to 10% of width, camber 0% to 9%
AI: average of 5 neural networks (2 hidden layers of 64), scikit-learn MLPRegressor
  inputs: 8 shape numbers + thickness + camber + camber position
Search (repeats): 400 random start wings, 6 rounds of 40 AI picks from 40,000 candidates,
  then 240 random wings for the fair comparison. Rejected wings get score 0 when training.
Wind-tunnel check: NACA 6409, Re = 61,400, data from Selig et al. 1995, Vol. 1
""")
    print(f"Made the folder 'share' with {n_scripts} scripts, {n_results} result tables, README, LICENSE,")
    print("requirements.txt and SETTINGS.txt. Your personal XFOIL path was removed from the scripts.")
    print("Open the folder and look through it before you upload anything.")

# ================================================================ main
COMMANDS = {"validation": cmd_validation, "summary": cmd_summary, "filters": cmd_filters,
            "reynolds": cmd_reynolds, "share": cmd_share}
if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    print("Please choose one:  " + "   ".join(COMMANDS))
else:
    COMMANDS[sys.argv[1]]()