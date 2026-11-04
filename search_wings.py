import os
import subprocess
import joblib
import numpy as np
import pandas as pd

XFOIL = r"PATH\TO\xfoil.exe"
RE = 40000

model = joblib.load("results/nn_model.joblib")
LOW = np.array([0.0, 0.2, 0.06])     # smallest m, p, t
HIGH = np.array([0.06, 0.6, 0.18])   # biggest m, p, t

def run_ga(seed, pop_size=300, rounds=100):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(LOW, HIGH, size=(pop_size, 3))
    for _ in range(rounds):
        score = model.predict(pop)
        order = np.argsort(score)[::-1]
        elite = pop[order[: pop_size // 5]]          # keep the best 20%
        kids = []
        while len(kids) < pop_size - len(elite):
            a = elite[rng.integers(len(elite))]
            b = elite[rng.integers(len(elite))]
            mix = rng.uniform(size=3)
            child = mix * a + (1 - mix) * b          # mix two good wings
            noise = rng.normal(0, 0.05, 3) * (HIGH - LOW)
            child = child + noise * (rng.random(3) < 0.3)   # small random change
            kids.append(np.clip(child, LOW, HIGH))   # stay inside allowed range
        pop = np.vstack([elite, kids])
    score = model.predict(pop)
    best = np.argmax(score)
    return pop[best], score[best]

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
    polar = f"polars/{name}.txt"
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

os.makedirs("data/wings", exist_ok=True)
os.makedirs("data/polars", exist_ok=True)

print("The AI is searching for the best wing...")
rows = []
for seed in range(5):
    (m, p, t), guess = run_ga(seed)
    name = f"ai_{seed:02d}"
    X, Y = naca4(m, p, t)
    with open(f"data/wings/{name}.dat", "w") as f:
        f.write(name + "\n")
        for xi, yi in zip(X, Y):
            f.write(f"{xi:.6f} {yi:.6f}\n")
    df = read_polar(run_xfoil(name))
    if df is None:
        real, n_ok, ang = None, 0, None
    else:
        best = df.loc[df["LD"].idxmax()]
        real, n_ok, ang = round(best["LD"], 2), len(df), best["alpha"]
    rows.append({"name": name, "m": round(m, 4), "p": round(p, 4),
                 "t": round(t, 4), "AI_guess": round(guess, 2),
                 "XFOIL_real": real, "n_ok": n_ok, "best_alpha": ang})

res = pd.DataFrame(rows)
res.to_csv("results/ai_search_check.csv", index=False)
print()
print(res.to_string(index=False))

old = pd.read_csv("results/results_re40000.csv")
old = old[old["n_ok"] >= 10]
print()
print("Best of the random wings:", old["best_LD"].max().round(2))
print("Your first wing (baseline):",
      old[old["name"] == "wing_000"]["best_LD"].iloc[0].round(2))