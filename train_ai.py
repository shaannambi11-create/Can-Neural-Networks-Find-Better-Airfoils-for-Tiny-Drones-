import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv("results/results_re40000.csv")
df = df[df["n_ok"] >= 10].dropna(subset=["best_LD"])
print("Wings used:", len(df))

X = df[["m", "p", "t"]].values
y = df["best_LD"].values

X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.50, random_state=42)
print("Learn / check / test:", len(y_train), len(y_val), len(y_test))
print()

models = {
    "Simple line (baseline)": make_pipeline(
        StandardScaler(), LinearRegression()),
    "Neural network": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000,
                     random_state=0)),
}

def report(model, X, y, label):
    guess = model.predict(X)
    r2 = r2_score(y, guess)
    miss = mean_absolute_error(y, guess)
    print(f"   {label:6s} R2 = {r2:.2f}   average miss = {miss:.2f}")

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name)
    report(model, X_train, y_train, "learn")
    report(model, X_val, y_val, "check")
    report(model, X_test, y_test, "test")
    print()

joblib.dump(models["Neural network"], "results/nn_model.joblib")
print("Saved the AI to results/nn_model.joblib")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, (name, model) in zip(ax, models.items()):
    guess = model.predict(X_test)
    a.scatter(y_test, guess, s=25)
    top = max(y_test.max(), guess.max()) + 2
    a.plot([0, top], [0, top], "r--")
    a.set_xlabel("real score (XFOIL)")
    a.set_ylabel("AI guess")
    a.set_title(name)
    a.grid(True)
plt.tight_layout()
plt.savefig("graphs/ai_vs_real.png", dpi=150)
plt.show()