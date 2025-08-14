# =========================================
# Minitab Model Comparison (Metrics Only)
# Matches your per-model pipelines/params
# =========================================
from mtbpy import mtbpy
import sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# Connect to Minitab
# ---------------------------
mtb = mtbpy.mtb_instance()

# ---------------------------
# Args
# ---------------------------
if len(sys.argv) < 3:
    mtb.add_message("Usage: python compare_models.py <target_col> <max_col> [--dt-depth N] [--dt-minsplit N] [--rf-maxdepth N]")
    sys.exit(1)

target_col = sys.argv[1]     # e.g., "C5"
max_col    = sys.argv[2]     # e.g., "C220"
max_idx    = int(max_col[1:])

# Optional flags (Decision Tree & Random Forest)
dt_depth = 4
dt_minsplit = 2
rf_maxdepth = 6
for i, arg in enumerate(sys.argv):
    if arg == "--dt-depth" and i + 1 < len(sys.argv):
        dt_depth = int(sys.argv[i + 1])
    elif arg == "--dt-minsplit" and i + 1 < len(sys.argv):
        dt_minsplit = int(sys.argv[i + 1])
    elif arg == "--rf-maxdepth" and i + 1 < len(sys.argv):
        rf_maxdepth = int(sys.argv[i + 1])

# ---------------------------
# Build column list & fetch data
# ---------------------------
all_columns  = [f"C{i}" for i in range(1, max_idx + 1)]
predict_cols = [c for c in all_columns if c != target_col]

y_series = pd.Series(mtb.get_column(target_col))
y_num = pd.to_numeric(y_series, errors="coerce")

X_raw = {col: pd.Series(mtb.get_column(col)) for col in predict_cols}
X_base = pd.DataFrame(X_raw).apply(pd.to_numeric, errors="coerce")

# Align X/y and drop rows with NaN anywhere
df_all = X_base.copy()
df_all["_y_"] = y_num.values
df_all = df_all.dropna(axis=0)

X_base = df_all.drop(columns=["_y_"])
y_base = df_all["_y_"].values
feature_names_base = X_base.columns.tolist()

# Common split settings (as in your scripts)
TEST_SIZE = 0.3
SEED = 42

# For clean metric printing
def _metrics(y_tr, y_tr_pred, y_te, y_te_pred):
    return (
        r2_score(y_tr, y_tr_pred),
        float(np.sqrt(mean_squared_error(y_tr, y_tr_pred))),
        r2_score(y_te, y_te_pred),
        float(np.sqrt(mean_squared_error(y_te, y_te_pred))),
    )

summary = []  # list of rows to print at end

# ==========================================================
# RIDGE (no scaling, no extra pruning) — your Ridge script
# ==========================================================
X_ridge = X_base.copy()
y_ridge = y_base.copy()
Xtr, Xte, ytr, yte = train_test_split(X_ridge, y_ridge, test_size=TEST_SIZE, random_state=SEED)

alphas_ridge = np.logspace(-3, 3, 50)
ridge = RidgeCV(alphas=alphas_ridge, cv=5)
ridge.fit(Xtr, ytr)
ytr_pred = ridge.predict(Xtr)
yte_pred = ridge.predict(Xte)

tr_r2, tr_rmse, te_r2, te_rmse = _metrics(ytr, ytr_pred, yte, yte_pred)
mtb.add_message(f"Model: Ridge Regression (alpha={ridge.alpha_:.4f})")
mtb.add_message(f"Target column: {target_col}")
mtb.add_message(f"Predictors used: {Xtr.shape[1]}")
mtb.add_message(f"Train R²: {tr_r2:.4f} | RMSE: {tr_rmse:.4f}")
mtb.add_message(f"Test  R²: {te_r2:.4f} | RMSE: {te_rmse:.4f}")
summary.append(("Ridge Regression", tr_r2, tr_rmse, te_r2, te_rmse))

# ==========================================================
# LASSO (prune + scale) — your LASSO script
# ==========================================================
X_lasso = X_base.copy()
# 1) remove near-zero variance
X_lasso = X_lasso.loc[:, X_lasso.var() > 1e-6]
# 2) remove highly correlated (|r| > 0.98)
corr = X_lasso.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if any(upper[c] > 0.98)]
if len(drop_cols) > 0:
    X_lasso = X_lasso.drop(columns=drop_cols)

# Align y and split
valid_cols_lasso = X_lasso.columns.tolist()
Xtr, Xte, ytr, yte = train_test_split(X_lasso, y_base[:len(X_lasso)], test_size=TEST_SIZE, random_state=SEED)

# Scale
scaler_lasso = StandardScaler()
Xtr_s = scaler_lasso.fit_transform(Xtr)
Xte_s = scaler_lasso.transform(Xte)

alphas_lasso = np.logspace(-2, 3, 80)
lasso = LassoCV(alphas=alphas_lasso, cv=5, random_state=SEED, max_iter=20000, tol=1e-4, selection='random')
lasso.fit(Xtr_s, ytr)
ytr_pred = lasso.predict(Xtr_s)
yte_pred = lasso.predict(Xte_s)

tr_r2, tr_rmse, te_r2, te_rmse = _metrics(ytr, ytr_pred, yte, yte_pred)
mtb.add_message(f"Model: LASSO Regression (alpha={lasso.alpha_:.6f})")
mtb.add_message(f"Target column: {target_col}")
mtb.add_message(f"Predictors used: {len(valid_cols_lasso)}")
mtb.add_message(f"Train R²: {tr_r2:.4f} | RMSE: {tr_rmse:.4f}")
mtb.add_message(f"Test  R²: {te_r2:.4f} | RMSE: {te_rmse:.4f}")
summary.append(("LASSO Regression", tr_r2, tr_rmse, te_r2, te_rmse))

# ==========================================================
# ELASTIC NET (prune + scale) — your Elastic Net script
# ==========================================================
X_en = X_base.copy()
# 1) remove near-zero variance
X_en = X_en.loc[:, X_en.var() > 1e-6]
# 2) remove highly correlated (|r| > 0.98)
corr_en = X_en.corr().abs()
upper_en = corr_en.where(np.triu(np.ones(corr_en.shape), k=1).astype(bool))
drop_cols_en = [c for c in upper_en.columns if any(upper_en[c] > 0.98)]
if len(drop_cols_en) > 0:
    X_en = X_en.drop(columns=drop_cols_en)

valid_cols_en = X_en.columns.tolist()
Xtr, Xte, ytr, yte = train_test_split(X_en, y_base[:len(X_en)], test_size=TEST_SIZE, random_state=SEED)

# Scale
scaler_en = StandardScaler()
Xtr_s = scaler_en.fit_transform(Xtr)
Xte_s = scaler_en.transform(Xte)

alphas_en = np.logspace(-2, 3, 80)
l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
en = ElasticNetCV(l1_ratio=l1_grid, alphas=alphas_en, cv=5, random_state=SEED, max_iter=20000, tol=1e-4)
en.fit(Xtr_s, ytr)
ytr_pred = en.predict(Xtr_s)
yte_pred = en.predict(Xte_s)

tr_r2, tr_rmse, te_r2, te_rmse = _metrics(ytr, ytr_pred, yte, yte_pred)
mtb.add_message(f"Model: Elastic Net (alpha={en.alpha_:.6f}, l1_ratio={en.l1_ratio_})")
mtb.add_message(f"Target column: {target_col}")
mtb.add_message(f"Predictors used: {len(valid_cols_en)}")
mtb.add_message(f"Train R²: {tr_r2:.4f} | RMSE: {tr_rmse:.4f}")
mtb.add_message(f"Test  R²: {te_r2:.4f} | RMSE: {te_rmse:.4f}")
summary.append(("Elastic Net", tr_r2, tr_rmse, te_r2, te_rmse))

# ==========================================================
# DECISION TREE (zero-var removal + scale) — your DT script
# ==========================================================
X_dt = X_base.copy()
X_dt = X_dt.loc[:, X_dt.var() > 1e-12]
valid_cols_dt = X_dt.columns.tolist()

Xtr, Xte, ytr, yte = train_test_split(X_dt, y_base[:len(X_dt)], test_size=TEST_SIZE, random_state=SEED)

# Scale (your DT script scales, even if not required)
scaler_dt = StandardScaler()
Xtr_s = scaler_dt.fit_transform(Xtr)
Xte_s = scaler_dt.transform(Xte)

dt = DecisionTreeRegressor(max_depth=dt_depth, min_samples_split=dt_minsplit, random_state=SEED)
dt.fit(Xtr_s, ytr)
ytr_pred = dt.predict(Xtr_s)
yte_pred = dt.predict(Xte_s)

tr_r2, tr_rmse, te_r2, te_rmse = _metrics(ytr, ytr_pred, yte, yte_pred)
mtb.add_message("Model: Decision Tree Regressor")
mtb.add_message(f"Parameters: Depth={dt_depth}, MinSplit={dt_minsplit}")
mtb.add_message(f"Target column: {target_col}")
mtb.add_message(f"Predictors used: {len(valid_cols_dt)}")
mtb.add_message(f"Train R²: {tr_r2:.4f} | RMSE: {tr_rmse:.4f}")
mtb.add_message(f"Test  R²: {te_r2:.4f} | RMSE: {te_rmse:.4f}")
summary.append((f"Decision Tree (depth={dt_depth}, minsplit={dt_minsplit})", tr_r2, tr_rmse, te_r2, te_rmse))

# ==========================================================
# RANDOM FOREST (no scaling) — your RF script
# ==========================================================
X_rf = X_base.copy()
valid_cols_rf = X_rf.columns.tolist()
Xtr, Xte, ytr, yte = train_test_split(X_rf, y_base[:len(X_rf)], test_size=TEST_SIZE, random_state=SEED)

rf = RandomForestRegressor(n_estimators=100, max_depth=rf_maxdepth, random_state=SEED)
rf.fit(Xtr, ytr)
ytr_pred = rf.predict(Xtr)
yte_pred = rf.predict(Xte)

tr_r2, tr_rmse, te_r2, te_rmse = _metrics(ytr, ytr_pred, yte, yte_pred)
mtb.add_message("Model: Random Forest Regressor")
mtb.add_message(f"Parameters: n_estimators=100, max_depth={rf_maxdepth}, random_state=42")
mtb.add_message(f"Target column: {target_col}")
mtb.add_message(f"Predictors used: {len(valid_cols_rf)}")
mtb.add_message(f"Train R²: {tr_r2:.4f} | RMSE: {tr_rmse:.4f}")
mtb.add_message(f"Test  R²: {te_r2:.4f} | RMSE: {te_rmse:.4f}")
summary.append((f"Random Forest (max_depth={rf_maxdepth})", tr_r2, tr_rmse, te_r2, te_rmse))

# ---------------------------
# Final summary (sorted by Test_R2 desc, Test_RMSE asc)
# ---------------------------
summary_df = pd.DataFrame(summary, columns=["Model", "Train_R2", "Train_RMSE", "Test_R2", "Test_RMSE"])
summary_df = summary_df.sort_values(by=["Test_R2", "Test_RMSE"], ascending=[False, True])

mtb.add_message("==== Model Comparison Summary ====")
for _, r in summary_df.iterrows():
    mtb.add_message(
        f"{r['Model']}: Train R²={r['Train_R2']:.4f}, Test R²={r['Test_R2']:.4f}, "
        f"Train RMSE={r['Train_RMSE']:.4f}, Test RMSE={r['Test_RMSE']:.4f}"
    )