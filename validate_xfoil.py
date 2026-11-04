import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- settings ----------
XFOIL = r"PATH\TO\xfoil.exe"
RE = 61400                    # Reynolds number of the published wind-tunnel test (Run 187)
NCRITS = [5, 9, 12]           # XFOIL "background disturbance" setting to try
AIRFOILS = {                  # name: (curve, curve position, thickness)
    "NACA 6409": (0.06, 0.4, 0.09),
    "NACA 0009": (0.0, 0.4, 0.09),
}

# ---------- numbers from the published UIUC wind-tunnel tests ----------
# Source: UIUC Low-Speed Airfoil Tests, Volume 1, Appendix B,
# NACA 6409 (Fig. 4.67), Run 187, Re = 61,400, clean surface.
# Each row is: (angle in degrees, CL, CD)
MEASURED = {
    "NACA 6409": [
        (-2.80, -0.005, 0.0628),
        (-1.14,  0.240, 0.0386),
        ( 0.26,  0.454, 0.0383),
        ( 2.07,  0.649, 0.0391),
        ( 3.32,  0.770, 0.0430),
        ( 4.98,  0.911, 0.0509),
    ],
    "NACA 0009": [],      # numbers not added yet
}

def naca4(m, p, t, n=80):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
              + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2),
                  m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu = x - yt*np.sin(th); yu = yc + yt*np.cos(th)
    xl = x + yt*np.sin(th); yl = yc - yt*np.cos(th)
    return (np.concatenate([xu[::-1], xl[1:]]),
            np.concatenate([yu[::-1], yl[1:]]))

os.makedirs("data/wings", exist_ok=True)
os.makedirs("data/polars_val", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

def run_xfoil(file_name, ncrit):
    polar = f"polars_val/{file_name}_n{ncrit}.txt"
    full = os.path.join("data", polar)
    if os.path.exists(full):
        os.remove(full)
    cmds = ["plop", "g", "", f"load wings/{file_name}.dat", "", "pane",
            "oper", f"visc {RE}", "vpar", f"n {ncrit}", "",
            "iter 200", "pacc", polar, "",
            "aseq -2 10 1", "pacc", "", "quit"]
    try:
        subprocess.run([XFOIL], input="\n".join(cmds) + "\n", text=True,
                       capture_output=True, cwd="data", timeout=120)
    except subprocess.TimeoutExpired:
        pass
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

results = {}
rows = []
for name, (m, p, t) in AIRFOILS.items():
    file_name = name.replace(" ", "").lower()
    X, Y = naca4(m, p, t)
    with open(f"data/wings/{file_name}.dat", "w") as f:
        f.write(name + "\n")
        for xi, yi in zip(X, Y):
            f.write(f"{xi:.6f} {yi:.6f}\n")
    for n in NCRITS:
        d = read_polar(run_xfoil(file_name, n))
        results[(name, n)] = d
        if d is None:
            print(f"{name}, setting n={n}: XFOIL gave no result")
            continue
        for _, r in d.iterrows():
            rows.append({"airfoil": name, "n": n, "alpha": r["alpha"],
                         "CL": r["CL"], "CD": r["CD"], "LD": r["LD"]})
        b = d.loc[d["LD"].idxmax()]
        print(f"{name}, n={n}: {len(d)}/13 angles ok, best L/D {b['LD']:.1f} at {b['alpha']:.0f} deg")

pd.DataFrame(rows).to_csv("results/xfoil_validation.csv", index=False)

print()
print("---------- XFOIL compared with the wind tunnel ----------")
for name, meas in MEASURED.items():
    if not meas:
        continue
    for n in NCRITS:
        d = results.get((name, n))
        if d is None:
            continue
        d = d.sort_values("alpha")
        print()
        print(f"{name}, XFOIL setting n={n}")
        print("   angle   CL tunnel   CL XFOIL   CD tunnel   CD XFOIL   CD difference")
        cl_diffs, cd_pct = [], []
        for a, cl_m, cd_m in meas:
            if a < d["alpha"].min() or a > d["alpha"].max():
                continue
            cl_x = float(np.interp(a, d["alpha"], d["CL"]))
            cd_x = float(np.interp(a, d["alpha"], d["CD"]))
            pct = 100 * (cd_x - cd_m) / cd_m
            cl_diffs.append(abs(cl_x - cl_m))
            cd_pct.append(abs(pct))
            print(f"   {a:5.2f}   {cl_m:8.3f}   {cl_x:8.3f}   {cd_m:8.4f}   {cd_x:8.4f}   {pct:+8.0f}%")
        if cl_diffs:
            print(f"   Average lift miss: {np.mean(cl_diffs):.3f}   "
                  f"Average drag miss: {np.mean(cd_pct):.0f}%")
print()
print("All XFOIL numbers saved in results\\xfoil_validation.csv")

fig, ax = plt.subplots(len(AIRFOILS), 3, figsize=(14, 4.2 * len(AIRFOILS)),
                       squeeze=False)
for r_i, name in enumerate(AIRFOILS):
    for n in NCRITS:
        d = results.get((name, n))
        if d is None:
            continue
        ax[r_i][0].plot(d["alpha"], d["CL"], "o-", ms=3, label=f"XFOIL n={n}")
        ax[r_i][1].plot(d["CD"], d["CL"], "o-", ms=3, label=f"XFOIL n={n}")
        ax[r_i][2].plot(d["alpha"], d["LD"], "o-", ms=3, label=f"XFOIL n={n}")
    meas = MEASURED.get(name, [])
    if meas:
        ax[r_i][0].plot([a for a, c, cd in meas], [c for a, c, cd in meas],
                        "ks", label="wind tunnel")
        ax[r_i][1].plot([cd for a, c, cd in meas], [c for a, c, cd in meas],
                        "ks", label="wind tunnel")
        ax[r_i][2].plot([a for a, c, cd in meas], [c / cd for a, c, cd in meas],
                        "ks", label="wind tunnel")
    ax[r_i][0].set_title(f"{name}: lift vs angle")
    ax[r_i][0].set_xlabel("angle (deg)"); ax[r_i][0].set_ylabel("CL")
    ax[r_i][1].set_title(f"{name}: lift vs drag")
    ax[r_i][1].set_xlabel("CD"); ax[r_i][1].set_ylabel("CL")
    ax[r_i][2].set_title(f"{name}: score vs angle")
    ax[r_i][2].set_xlabel("angle (deg)"); ax[r_i][2].set_ylabel("L/D")
    for a in ax[r_i]:
        a.grid(True)
        a.legend(fontsize=8)
plt.tight_layout()
plt.savefig("graphs/xfoil_vs_uiuc.png", dpi=150)
print("Graph saved: graphs\\xfoil_vs_uiuc.png")
plt.show()