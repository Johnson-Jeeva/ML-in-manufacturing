# minitab_models_compare.py
import sys, warnings, numpy as np, pandas as pd
from mtbpy import mtbpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.base import clone

warnings.filterwarnings("ignore")
mtb = mtbpy.mtb_instance()

# ---------------------------
# Args: target column like C5 and max column like C220
# ---------------------------
target_col = sys.argv[1]     # e.g., C5  (this is your TARGET)
max_col    = sys.argv[2]     # e.g., C220 (scan C1..C220)
TARGET_NAME = "Strength2"    # label for output only

# Build column list and predictors
max_idx = int(max_col[1:])
all_columns = [f"C{i}" for i in range(1, max_idx + 1)]
predictor_cols = [c for c in all_columns if c != target_col]

# ---------------------------
# Pull data from Minitab
# ---------------------------
def get_series(col):
    try:
        return np.array(mtb.get_column(col), dtype=float)
    except Exception:
        return None

y_full = get_series(target_col)
if y_full is None:
    mtb.add_message(f"[Error] Target {target_col} not found.")
    sys.exit(1)

X_data, valid_cols = [], []
for col in predictor_cols:
    s = get_series(col)
    if s is not None:
        X_data.append(s)
        valid_cols.append(col)
    else:
        mtb.add_message(f"Warning: {col} missing/non-numeric. Skipping.")

if len(valid_cols) == 0:
    mtb.add_message("[Error] No valid predictor columns.")
    sys.exit(1)

# Assemble dataframe and clean
X_np = np.vstack(X_data).T  # shape (n_rows, n_features)
df = pd.DataFrame(X_np, columns=valid_cols)
# Drop rows with any NaNs in X or y
df["__y__"] = y_full[:len(df)]
df = df.dropna(subset=df.columns.tolist())
y = df["__y__"].values
X_df = df.drop(columns="__y__")

valid_cols = X_df.columns.tolist()

# ---------------------------
# Train/Validation Split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_df, y, test_size=0.30, random_state=42
)

rmse = lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))

# ---------------------------
# Pipelines (mirror your JMP setup)
# ---------------------------
ols   = Pipeline([("scaler", "passthrough"), ("model", LinearRegression())])
ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))])
lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=10000, random_state=42))])
enet  = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42))])

alphas = np.logspace(-3, 1, 8)
l1s    = [0.2, 0.5, 0.8]

ridge_cv = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=alphas, cv=3))])
lasso_cv = Pipeline([("scaler", StandardScaler()), ("model", LassoCV(alphas=alphas, cv=3, max_iter=5000, random_state=42))])
enet_cv  = Pipeline([("scaler", StandardScaler()), ("model", ElasticNetCV(alphas=alphas, l1_ratio=l1s, cv=3, max_iter=5000, random_state=42))])

models = [
    ("OLS", ols),
    ("Ridge", ridge),
    ("Lasso", lasso),
    ("ElasticNet", enet),
    ("RidgeCV", ridge_cv),
    ("LassoCV", lasso_cv),
    ("ElasticNetCV", enet_cv),
]

# ---------------------------
# Fit, evaluate, collect metrics
# ---------------------------
rows = []
for name, pipe in models:
    pipe.fit(X_train, y_train)
    ytr = pipe.predict(X_train)
    yva = pipe.predict(X_val)
    rows.append({
        "Model": name,
        "Train R2": r2_score(y_train, ytr),
        "Val R2":   r2_score(y_val,   yva),
        "Train RMSE": rmse(y_train, ytr),
        "Val RMSE":   rmse(y_val,   yva),
    })

metrics = pd.DataFrame(rows).sort_values("Val RMSE").reset_index(drop=True)

mtb.add_message("\n=== Model Comparison (sorted by Val RMSE) ===")
mtb.add_message(metrics.to_string(index=False))

# ---------------------------
# OLS summary (statsmodels)
# ---------------------------
try:
    import statsmodels.api as sm
    mtb.add_message("\n=== OLS Summary ===")
    sm.OLS(y_train, sm.add_constant(X_train)).fit().summary().print_summary()
except Exception as e:
    mtb.add_message(f"\n[Info] OLS summary skipped: {e}")

# ---------------------------
# Standardized coefficients + compact equation text
# ---------------------------
from sklearn.preprocessing import StandardScaler
def std_coefs(pipe):
    pipe.fit(X_train, y_train)
    scaler = pipe.named_steps.get("scaler", None)
    model  = pipe.named_steps["model"]
    if isinstance(scaler, StandardScaler):
        coefs = pd.Series(model.coef_, index=X_train.columns)
        intercept = float(model.intercept_)
    else:
        p = Pipeline([("scaler", StandardScaler()), ("model", clone(model))])
        p.fit(X_train, y_train)
        coefs = pd.Series(p.named_steps["model"].coef_, index=X_train.columns)
        intercept = float(p.named_steps["model"].intercept_)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    eq = "y = {:.4f} + ".format(intercept) + " + ".join([f"({c:.4f} * {f})" for f, c in coefs.items()])
    return coefs, intercept, eq

mtb.add_message("\n=== Coefficients & Equations (standardized) ===")
for name, pipe in models:
    c, b0, eq = std_coefs(pipe)
    mtb.add_message(f"\n--- {name} ---")
    mtb.add_message(f"Intercept: {b0:.6f}")
    mtb.add_message("Top 10 Coefficients:\n" + c.head(10).to_string())
    mtb.add_message("Equation (truncated):\n" + (eq[:400] + " ...") if len(eq) > 400 else eq)

# ---------------------------
# Build full-length prediction columns aligned to original rows
# ---------------------------
n_rows = len(df)  # after cleaning/alignment
index_all = df.index  # keep alignment

res = pd.DataFrame(index=index_all)
res["Actual"] = y
res["Split"]  = "Unassigned"
res.loc[X_train.index, "Split"] = "Training"
res.loc[X_val.index,   "Split"] = "Validation"

for name, pipe in models:
    yhat = pd.Series(np.nan, index=index_all)
    pipe.fit(X_train, y_train)
    yhat.loc[X_train.index] = pipe.predict(X_train)
    yhat.loc[X_val.index]   = pipe.predict(X_val)
    res[f"{name}_Pred"] = yhat.values

# ---------------------------
# Write columns back to Minitab worksheet
# ---------------------------
def put_col(name, values):
    mtb.put_column(name, list(values))

put_col("Actual", res["Actual"].values)
put_col("Split",  res["Split"].astype(str).values)

for col in res.columns:
    if col.endswith("_Pred"):
        put_col(col, res[col].values)

mtb.add_message(f"\nWrote columns: {', '.join(['Actual','Split'] + [c for c in res.columns if c.endswith('_Pred')])}")