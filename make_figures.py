import os
from math import comb
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---------- settings ----------
LASTNAME = "nambi"            # used in the file names (change if needed)
OUT = "graphs/paper"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
BLUE, GRAY, RED = "#1f5fa8", "#8a8a8a", "#c0392b"

# ---------- published wind-tunnel data (UIUC Vol. 1, NACA 6409, Re = 61,400) ----------
MEASURED = [(-2.80, -0.005, 0.0628), (-1.14, 0.240, 0.0386), (0.26, 0.454, 0.0383),
            (2.07, 0.649, 0.0391), (3.32, 0.770, 0.0430), (4.98, 0.911, 0.0509)]

WCOLS = [f"w{i}" for i in range(8)]

def surface(x, w):
    n = len(w) - 1
    K_ = sum(w[i] * comb(n, i) * x**i * (1 - x)**(n - i) for i in range(n + 1))
    return np.sqrt(x) * (1 - x) * K_

def naca4(m, p, t, n=300):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
              + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2),
                  m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    return x - yt*np.sin(th), yc + yt*np.cos(th), x + yt*np.sin(th), yc - yt*np.cos(th)

def usable(df):
    return df[(df["n_ok"] >= 10) & df["best_LD"].notna()]

def load(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def save(fig, name):
    path = f"{OUT}/{LASTNAME}_{name}.jpg"
    fig.savefig(path, dpi=600, pil_kwargs={"quality": 95})
    plt.close(fig)
    print("saved", path)

# ---------- load data ----------
import glob, re as _re
_nums = sorted(int(_re.findall(r"rep(\d+)", os.path.basename(f))[0]) for f in glob.glob("results/rep[0-9]*.csv") if "_test" not in f)
reps = {i: load(f"results/rep{i}.csv") for i in _nums}
big = load("results/cst_data.csv")
std = load("results/results_re20000.csv")
val = load("results/xfoil_validation.csv")

baseline = 3.95
if std is not None and (std["name"] == "wing_000").any():
    baseline = float(std[std["name"] == "wing_000"]["best_LD"].iloc[0])

frames = [d for d in [big] + list(reps.values()) if d is not None]
allw = usable(pd.concat(frames, ignore_index=True))
best = allw.loc[allw["best_LD"].idxmax()]
bw = best[WCOLS].values.astype(float)
captions = []

# ---------- Figure 1: wing shapes ----------
xs = np.linspace(0, 1, 300)
xu, yu, xl, yl = naca4(0.02, 0.4, 0.12)
fig, axs = plt.subplots(2, 1, figsize=(3.3, 1.9))
for ax, (x_u, y_u, x_l, y_l), colr, label in [
        (axs[0], (xu, yu, xl, yl), RED,
         f"Standard NACA 2412: score {baseline:.2f}, 12% thick"),
        (axs[1], (xs, surface(xs, bw[:4]), xs, surface(xs, bw[4:])), BLUE,
         f"Best AI-guided wing: score {best['best_LD']:.1f}, "
         f"{100*best['t']:.1f}% thick")]:
    ax.fill(np.concatenate([x_u, x_l[::-1]]), np.concatenate([y_u, y_l[::-1]]),
            color=colr, alpha=0.25, lw=0)
    ax.plot(np.concatenate([x_u, x_l[::-1]]), np.concatenate([y_u, y_l[::-1]]),
            color=colr, lw=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.06, 0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.0, 0.125, label, ha="left", va="center")
save(fig, "fig01")
where = "the first large run" if not str(best["name"]).startswith("s") else f"repeat {str(best['name'])[1]}"
captions.append(f"Figure 1. Cross-sections of a standard NACA 2412 wing and the best wing found by the "
                f"AI-guided search ({best['name']}, from {where}), drawn to the same scale. Scores are lift-to-drag "
                f"ratios at a Reynolds number of 20,000 (higher is better).")

# ---------- Figure 2: thickness vs score ----------
fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.scatter(100 * allw["t"], allw["best_LD"], s=3, color=GRAY, alpha=0.35, lw=0,
           label="Wings tested (passed checks)")
ax.scatter([12], [baseline], s=28, color=RED, marker="s", zorder=3,
           label="Standard NACA 2412")
ax.axvspan(2, 10, color="#e8eef7", zorder=0, label="Range searched (2 to 10%)")
ax.set_xlabel("Wing thickness (% of width)")
ax.set_ylabel("Score (lift-to-drag ratio)")
ax.set_xlim(1.5, 13)
ax.set_ylim(0, None)
ax.legend(frameon=False, loc="upper right", markerscale=2.2, fontsize=7.5)
fig.tight_layout(pad=0.4)
save(fig, "fig02")
captions.append(f"Figure 2. Score versus thickness for {len(allw)} wings that passed the quality checks "
                f"at a Reynolds number of 20,000. Thinner wings scored higher. The standard NACA 2412 "
                f"wing (red) is 12% thick, thicker than any wing in the range searched (shaded).")

# ---------- Figure 3: AI vs random ----------
def compare(df):
    good = usable(df)
    ai = good[good["source"].isin(["random", "ai"])].sort_values("order")
    rd = good[good["source"].isin(["random", "random_arm"])].sort_values("order")
    n = min(len(ai), len(rd))
    return n, ai["best_LD"].head(n).max(), rd["best_LD"].head(n).max()

rows = []
for i, d in reps.items():
    if d is not None and (d["source"] == "random_arm").any():
        n, a, r = compare(d)
        rows.append((i, n, a, r))
if rows:
    from matplotlib.lines import Line2D
    ORANGE = "#e08e0b"
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for i, n, a, r in rows:
        col = BLUE if a > r else (GRAY if a == r else ORANGE)
        ax.plot([0, 1], [r, a], "-o", color=col, ms=3, lw=0.9, alpha=0.8, zorder=2)
    ra = np.mean([r[3] for r in rows]); aa = np.mean([r[2] for r in rows])
    ax.plot([0, 1], [ra, aa], "-s", color="black", ms=5, lw=2.0, zorder=3)
    wins = sum(1 for r in rows if r[2] > r[3])
    ties = sum(1 for r in rows if r[2] == r[3])
    loss = sum(1 for r in rows if r[2] < r[3])
    handles = [Line2D([0], [0], color=BLUE, lw=1.2, label=f"AI better ({wins})"),
               Line2D([0], [0], color=GRAY, lw=1.2, label=f"Tie ({ties})"),
               Line2D([0], [0], color=ORANGE, lw=1.2, label=f"AI worse ({loss})"),
               Line2D([0], [0], color="black", lw=2.0, marker="s", ms=4, label="Average")]
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random\nsearch", "AI-guided\nsearch"])
    ax.set_ylabel("Best score found")
    vals = [v for r in rows for v in (r[2], r[3])]
    lo, hi = min(vals), max(vals)
    ax.set_ylim(lo - 1.5, hi + 0.5 * max(hi - lo, 3) + 1.5)
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2, fontsize=7.5, columnspacing=0.8)
    fig.tight_layout(pad=0.4)
    save(fig, "fig03")
    diffs = [r[2] - r[3] for r in rows]
    captions.append(f"Figure 3. Best score found by random and AI-guided search in {len(rows)} repeats, "
                    f"with equal numbers of successful tests. Each line is one repeat. The AI-guided search was "
                    f"better in {wins}, tied in {ties}, and worse in {loss}; the average difference was "
                    f"{np.mean(diffs):+.1f} score units. The standard NACA 2412 wing scored {baseline:.2f}, far below this range.")

# ---------- Figure 4: XFOIL vs wind tunnel ----------
if val is not None:
    v = val[val["airfoil"] == "NACA 6409"]
    fig, axs = plt.subplots(2, 1, figsize=(3.3, 3.5))
    first = True
    for n, style in [(5, "other"), (12, "other"), (9, "main")]:
        d = v[v["n"] == n].sort_values("alpha")
        d = d[d["alpha"] <= 6]          # wind-tunnel data stop near 5 degrees
        if len(d) == 0:
            continue
        for ax, xcol, ycol in [(axs[0], "alpha", "CL"), (axs[1], "CD", "CL")]:
            if style == "main":
                ax.plot(d[xcol], d[ycol], color=BLUE, label="XFOIL, normal setting (n = 9)")
            else:
                ax.plot(d[xcol], d[ycol], color="#b5b5b5", ls="--", lw=0.8,
                        label="XFOIL, other settings (n = 5, 12)" if (first and n == 5) else None)
        first = False if n == 5 else first
    axs[0].plot([m[0] for m in MEASURED], [m[1] for m in MEASURED], "ks", ms=3.5,
                label="Wind tunnel")
    axs[1].plot([m[2] for m in MEASURED], [m[1] for m in MEASURED], "ks", ms=3.5)
    axs[0].set_xlabel("Angle (degrees)")
    axs[0].set_ylabel("Lift coefficient")
    axs[1].set_xlabel("Drag coefficient")
    axs[1].set_ylabel("Lift coefficient")
    axs[0].set_xlim(-3.5, 7)
    _cds = [m[2] for m in MEASURED] + list(v[v["alpha"] <= 6]["CD"])
    axs[1].set_xlim(min(_cds) - 0.003, max(_cds) + 0.003)
    axs[0].set_ylim(None, axs[0].get_ylim()[1] + 0.75)
    axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.11, 1.0))
    axs[0].text(0.02, 0.94, "(a)", transform=axs[0].transAxes, fontweight="bold")
    axs[1].text(0.02, 0.94, "(b)", transform=axs[1].transAxes, fontweight="bold")
    fig.tight_layout(pad=0.4)
    save(fig, "fig04")
    d9 = v[(v["n"] == 9) & (v["alpha"] <= 6)].sort_values("alpha")
    clm, cdm = [], []
    for a, cl_m, cd_m in MEASURED:
        if d9["alpha"].min() <= a <= d9["alpha"].max():
            clm.append(abs(np.interp(a, d9["alpha"], d9["CL"]) - cl_m))
            cdm.append(abs(100 * (np.interp(a, d9["alpha"], d9["CD"]) - cd_m) / cd_m))
    captions.append(f"Figure 4. XFOIL compared with published wind-tunnel measurements of a NACA 6409 wing "
                    f"at a Reynolds number of 61,400. (a) Lift versus angle. (b) Lift versus drag. XFOIL is shown up to 6 degrees, since the wind-tunnel data reach about 5 degrees. "
                    f"With the normal setting, lift was off by {np.mean(clm):.3f} and drag by "
                    f"{np.mean(cdm):.0f}% on average.")

# ---------- Table of Contents image (max 7.0 cm x 3.5 cm) ----------
fig = plt.figure(figsize=(2.7, 1.35))
for k, ((x_u, y_u, x_l, y_l), colr, text) in enumerate([
        ((xu, yu, xl, yl), RED, f"Standard wing\nscore {baseline:.1f}"),
        ((xs, surface(xs, bw[:4]), xs, surface(xs, bw[4:])), BLUE,
         f"AI-guided wing\nscore {best['best_LD']:.0f}")]):
    ax = fig.add_axes([0.02, 0.52 - 0.48 * k, 0.60, 0.46])
    ax.fill(np.concatenate([x_u, x_l[::-1]]), np.concatenate([y_u, y_l[::-1]]),
            color=colr, alpha=0.25, lw=0)
    ax.plot(np.concatenate([x_u, x_l[::-1]]), np.concatenate([y_u, y_l[::-1]]),
            color=colr, lw=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.06, 0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.66, 0.74 - 0.48 * k, text, color=colr, ha="left", va="center")
save(fig, "TOC")

with open(f"{OUT}/captions.txt", "w") as f:
    f.write("\n\n".join(captions))
print()
print("Suggested captions saved in", f"{OUT}/captions.txt")