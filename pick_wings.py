import os
import glob
from math import comb
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- settings ----------
CHORD_CM = 10          # how wide the test wing will be (front to back)
HOW_MANY = 3           # how many different best wings to pick
RESULT_FILES = ["results/cst_data.csv"] + sorted(glob.glob("results/rep[0-9]*.csv"))

WCOLS = [f"w{i}" for i in range(8)]
UP = (0.02, 0.35)
LO = (-0.30, 0.15)
LOW = np.array([UP[0]] * 4 + [LO[0]] * 4)
HIGH = np.array([UP[1]] * 4 + [LO[1]] * 4)

def surface(x, w):
    n = len(w) - 1
    K_ = sum(w[i] * comb(n, i) * x**i * (1 - x)**(n - i) for i in range(n + 1))
    return np.sqrt(x) * (1 - x) * K_

def naca4(m, p, t, n=200):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
              + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2),
                  m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    return (x - yt*np.sin(th), yc + yt*np.cos(th),
            x + yt*np.sin(th), yc - yt*np.cos(th))

# ---------- load all results ----------
frames = []
for f in RESULT_FILES:
    if os.path.exists(f) and "_test" not in f and "before_filter" not in f:
        d = pd.read_csv(f)
        d["file"] = os.path.basename(f)
        frames.append(d)
df = pd.concat(frames, ignore_index=True)
good = df[(df["n_ok"] >= 10) & df["best_LD"].notna()].copy()
good = good.sort_values("best_LD", ascending=False)
print("Good wings found in all files:", len(good))

# ---------- pick the best wings that look different from each other ----------
scale = HIGH - LOW
chosen = []
for _, row in good.iterrows():
    w = row[WCOLS].values.astype(float)
    if all(np.max(np.abs(w - c[WCOLS].values.astype(float)) / scale) > 0.10
           for c in chosen):
        chosen.append(row)
    if len(chosen) == HOW_MANY:
        break

print()
print(f"Best {HOW_MANY} different wings (wing width {CHORD_CM} cm):")
for row in chosen:
    print(f"  {row['name']:12s} from {row['file']:14s} score {row['best_LD']:.2f}"
          f"  best angle {row['best_alpha']:.0f} deg"
          f"  thickness {100*row['t']:.1f}% = {10*CHORD_CM*row['t']:.1f} mm"
          f"  curve {100*row['m']:.1f}%  curve position {row['p']:.2f}"
          f"  good angles {int(row['n_ok'])}/13")

# ---------- actual-size drawings ----------
os.makedirs("results", exist_ok=True)
page_w, page_h = 11.69, 8.27          # A4 sideways, in inches
inch = CHORD_CM / 2.54                # length of the wing on paper, in inches

def draw_page(pdf, title, lines, x, yu, yl):
    fig = plt.figure(figsize=(page_w, page_h))
    xr, yr = (-0.1, 1.1), (-0.16, 0.16)
    aw = (xr[1] - xr[0]) * inch
    ah = (yr[1] - yr[0]) * inch
    ax = fig.add_axes([(page_w - aw) / 2 / page_w, 0.42,
                       aw / page_w, ah / page_h])
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.plot(np.concatenate([x, x[::-1]]), np.concatenate([yu, yl[::-1]]), "k", lw=1.2)
    ax.plot([0, 1], [-0.13, -0.13], "r", lw=1)
    for cm in range(CHORD_CM + 1):
        ax.plot([cm / CHORD_CM] * 2, [-0.13, -0.115], "r", lw=1)
    ax.text(0.5, -0.215, f"This red line must measure exactly {CHORD_CM} cm. "
            "Print at 100% size.", ha="center", fontsize=10, color="r")
    ax.axis("off")
    fig.text(0.5, 0.92, title, ha="center", fontsize=16)
    for i, t in enumerate(lines):
        fig.text(0.5, 0.86 - 0.04 * i, t, ha="center", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

xs = np.linspace(0, 1, 300)
with PdfPages("results/wing_templates.pdf") as pdf:
    for k, row in enumerate(chosen, 1):
        w = row[WCOLS].values.astype(float)
        draw_page(pdf, f"Wing {k}: {row['name']}  (AI-found)",
                  [f"Score {row['best_LD']:.2f}, best angle {row['best_alpha']:.0f} deg",
                   f"Thickness {10*CHORD_CM*row['t']:.1f} mm at {CHORD_CM} cm width"],
                  xs, surface(xs, w[:4]), surface(xs, w[4:]))
    xb = np.linspace(0, 1, 300)
    xu, yu_, xl, yl_ = naca4(0.02, 0.4, 0.12)
    yu_i = np.interp(xb, xu, yu_)
    yl_i = np.interp(xb, xl, yl_)
    draw_page(pdf, "Comparison wing: NACA 2412 (standard wing)",
              [f"Thickness {10*CHORD_CM*0.12:.1f} mm at {CHORD_CM} cm width"],
              xb, yu_i, yl_i)
print()
print("Drawings saved: results\\wing_templates.pdf")