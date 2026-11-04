import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 500
rng = np.random.default_rng(42)

def naca4(m, p, t, n=80):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
              + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p,
                  m/p**2*(2*p*x - x**2),
                  m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p,
                   2*m/p**2*(p - x),
                   2*m/(1-p)**2*(p - x))
    theta = np.arctan(dyc)
    xu = x - yt*np.sin(theta); yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta); yl = yc - yt*np.cos(theta)
    X = np.concatenate([xu[::-1], xl[1:]])
    Y = np.concatenate([yu[::-1], yl[1:]])
    return X, Y

params = [("wing_000", 0.02, 0.4, 0.12)]
for i in range(1, N):
    m = round(rng.uniform(0.0, 0.06), 4)
    p = round(rng.uniform(0.2, 0.6), 4)
    t = round(rng.uniform(0.06, 0.18), 4)
    params.append((f"wing_{i:03d}", m, p, t))

os.makedirs("data/wings", exist_ok=True)
for name, m, p, t in params:
    X, Y = naca4(m, p, t)
    with open(f"data/wings/{name}.dat", "w") as f:
        f.write(name + "\n")
        for xi, yi in zip(X, Y):
            f.write(f"{xi:.6f} {yi:.6f}\n")

pd.DataFrame(params, columns=["name", "m", "p", "t"]).to_csv(
    "data/wing_params.csv", index=False)
print("Made", len(params), "wings")

fig, axes = plt.subplots(2, 3, figsize=(12, 5))
for ax, (name, m, p, t) in zip(axes.flat, params[:6]):
    X, Y = naca4(m, p, t)
    ax.plot(X, Y)
    ax.set_aspect("equal")
    ax.set_title(f"{name}  m={m} p={p} t={t}", fontsize=9)
plt.tight_layout()
plt.savefig("graphs/sample_wings.png", dpi=150)
plt.show()