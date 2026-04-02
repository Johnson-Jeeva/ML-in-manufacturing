# =============================================================================
# Minitab Python Integration — Hybrid Predictive Modelling Script
# Model  : Stacking Ensemble (Deep ANN + Random Forest)
#          Uses a meta-learner to optimally combine ANN and Tree predictions
# Output : Performance metrics, ensemble weights, permutation feature importance, 
#          ANN loss curve, actual vs predicted, residual diagnostics, SHAP, 
#          PDP, ICE, learning curves — all displayed in the Minitab output pane
#
# WHY A HYBRID MODEL:
#   Combines the deep feature extraction and non-linear capabilities of ANNs
#   with the robust, outlier-resistant partitioning of Tree-based models. 
#   A Ridge regression meta-learner dynamically calculates the optimal weighted 
#   average of both base models for maximum accuracy.
#
# Arguments (passed from Minitab):
#   Training mode : <target_col> <max_predictor_col>
#                   e.g.  C5 C4   →  target=C5, predictors=C1–C4
#   Predict mode  : predict <target_col> <max_predictor_col> <values>
#                   e.g.  predict C5 C4 1.2,3.4,5.6,7.8
# =============================================================================

# -----------------------------------------------------------------------------
# Must be first — blocks tkinter/TkAgg from loading in Minitab's bg thread
# -----------------------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from mtbpy import mtbpy

# Sklearn imports
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
import gc

# =============================================================================
# SCALING CONSTANTS  — tuned for speed without sacrificing output quality
# =============================================================================
MAX_FEATURES_PLOT    = 20    # max bars in importance chart
MAX_FEATURES_TABLE   = 25    # max rows in importance table
PERM_IMP_REPEATS     =  8    # permutation importance repeats
SHAP_MAX_ROWS        = 150   # validation rows sent to KernelExplainer
SHAP_BACKGROUND_ROWS =  50   # kmeans background points
SHAP_MAX_DISPLAY     = 20    # features shown in beeswarm / waterfall
PDP_TOP_N_FEATURES   =  3    # features in PDP panel
ICE_MAX_LINES        = 100   # individual ICE lines
SCATTER_MAX_POINTS   = 800   # points in scatter / residual plots

# =============================================================================

# -----------------------------------------------------------------------------
# 1. CONNECT TO MINITAB
# -----------------------------------------------------------------------------
mtb = mtbpy.mtb_instance()

# -----------------------------------------------------------------------------
# 2. PLOTS FOLDER
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir  = os.path.join(script_dir, "Plots_Hybrid_Fast")
os.makedirs(plots_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 3. PARSE ARGUMENTS
# -----------------------------------------------------------------------------
args         = sys.argv[1:]
predict_mode = args[0].lower() == "predict"

if predict_mode:
    _, target_col, max_col, input_str = args
    input_values = [float(x.strip()) for x in input_str.split(",")]
else:
    target_col, max_col = args

# -----------------------------------------------------------------------------
# 4. PREDICTOR COLUMN LIST
# -----------------------------------------------------------------------------
max_idx        = int(max_col[1:])
predictor_cols = (
    [f"C{i}" for i in range(1, max_idx + 1) if f"C{i}" != target_col]
    if not predict_mode
    else [f"C{i}" for i in range(1, len(input_values) + 1)]
)

# -----------------------------------------------------------------------------
# 5. LOAD PREDICTORS
# -----------------------------------------------------------------------------
X_data, valid_cols = [], []
for col in predictor_cols:
    try:
        X_data.append(mtb.get_column(col))
        valid_cols.append(col)
    except:
        mtb.add_message(f"Warning: Column {col} not found — skipping.")

X_df           = pd.DataFrame(np.array(X_data).T, columns=valid_cols).dropna()
final_features = X_df.columns.tolist()

# -----------------------------------------------------------------------------
# 6. LOAD TARGET
# -----------------------------------------------------------------------------
try:
    y = np.array(mtb.get_column(target_col))[:len(X_df)]
except:
    mtb.add_message(f"Error: Target column '{target_col}' not found.")
    sys.exit()

# -----------------------------------------------------------------------------
# 7. BUILD THE BEST HYBRID MODEL
#
#    Base Model 1 : Deep ANN (Pyramid) + StandardScaler
#    Base Model 2 : Random Forest Regressor (Handles unscaled/noisy data well)
#    Meta-Learner : RidgeCV (Finds optimal weighted average of Model 1 & 2)
# -----------------------------------------------------------------------------

def build_model(n_features, n_rows):
    # --- ANN Setup ---
    width  = int(np.clip(n_features * 4, 64, 512))
    mid    = max(32, width // 2)
    narrow = max(16, width // 4)
    batch  = min(256, max(32, n_rows // 5))

    mlp = MLPRegressor(
        hidden_layer_sizes  = (width, mid, narrow),
        activation          = "relu",
        solver              = "adam",
        alpha               = 1e-3,
        learning_rate       = "adaptive",
        learning_rate_init  = 0.001,
        max_iter            = 2000,
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 40,
        batch_size          = batch,
        tol                 = 1e-5,
        random_state        = 42,
    )
    ann_pipeline = Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])
    
    # --- Tree Setup ---
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=max(5, n_features // 2 + 2),
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # RF utilizes all cores here
    )
    
    # --- Meta-Learner Stacker Setup ---
    stacker = StackingRegressor(
        estimators=[
            ('ann', ann_pipeline),
            ('rf', rf)
        ],
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=1 # Prevents core-thrashing since RF already uses n_jobs=-1
    )
    
    return stacker

# -----------------------------------------------------------------------------
# 8. PREDICT MODE
# -----------------------------------------------------------------------------
if predict_mode:
    X_df = X_df.loc[:, X_df.var() > 1e-6]
    final_features = X_df.columns.tolist()
    y = y[:len(X_df)]

    model = build_model(len(final_features), len(X_df))
    model.fit(X_df, y)

    input_df   = pd.DataFrame([input_values], columns=final_features)
    prediction = model.predict(input_df)[0]
    mtb.add_message(
        f"Predicted Value : {prediction:.4f}\n"
        f"Model           : Hybrid Ensemble (ANN + Random Forest)\n"
        f"Target          : {target_col}"
    )
    sys.exit()

# -----------------------------------------------------------------------------
# 9. DATA CLEANING
# -----------------------------------------------------------------------------
X_df = X_df.loc[:, X_df.var() > 1e-6]

corr_matrix = X_df.corr().abs()
upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols   = [col for col in upper.columns if any(upper[col] > 0.98)]
if drop_cols:
    X_df = X_df.drop(columns=drop_cols)
    mtb.add_message(
        f"Data Cleaning: Removed {len(drop_cols)} highly correlated "
        f"predictor(s): {', '.join(drop_cols)}"
    )

y              = y[:len(X_df)]
final_features = X_df.columns.tolist()
n_features     = len(final_features)
n_rows         = len(X_df)

mtb.add_message(
    f"Dataset : {n_rows} rows  |  {n_features} predictors  |  Target: {target_col}\n"
    f"Model   : Hybrid Stacking Ensemble\n"
    f"          - Base 1: Deep ANN ({int(np.clip(n_features*4,64,512))} "
    f"→ {max(32,int(np.clip(n_features*4,64,512))//2)} "
    f"→ {max(16,int(np.clip(n_features*4,64,512))//4)}) neurons\n"
    f"          - Base 2: Random Forest\n"
    f"          - Meta  : RidgeCV weighted average"
)

# -----------------------------------------------------------------------------
# 10. TRAIN / VALIDATION SPLIT  (70 / 30)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=42
)
X_train_df = pd.DataFrame(X_train, columns=final_features)
X_test_df  = pd.DataFrame(X_test,  columns=final_features)

# -----------------------------------------------------------------------------
# 11. TRAIN
# -----------------------------------------------------------------------------
model = build_model(n_features, n_rows)
model.fit(X_train_df, y_train)

# Extract internal components for reporting
ann_step     = model.named_estimators_['ann'].named_steps['mlp']
rf_step      = model.named_estimators_['rf']
meta_weights = model.final_estimator_.coef_

y_train_pred = model.predict(X_train_df)
y_test_pred  = model.predict(X_test_df)
residuals    = y_test - y_test_pred

r2_train   = r2_score(y_train, y_train_pred)
r2_test    = r2_score(y_test,  y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
mad        = np.mean(np.abs(residuals))
n_iter     = getattr(ann_step, "n_iter_", "N/A")

arch = ann_step.hidden_layer_sizes
arch_str = " -> ".join(str(x) for x in arch) if isinstance(arch, tuple) else str(arch)

# Calculate relative reliance (weights) of the meta learner
# Softmax logic to convert Ridge coefficients into rough % reliance 
exp_weights = np.exp(meta_weights)
rel_weights = exp_weights / np.sum(exp_weights) * 100
ann_reliance = rel_weights[0]
rf_reliance  = rel_weights[1]

# =============================================================================
# HELPERS
# =============================================================================

def save_and_push(fig, filename):
    fpath = os.path.join(plots_dir, filename)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    mtb.add_image(fpath)
    plt.close(fig)

def bar_chart_height(n):
    return max(4, min(20, n * 0.45 + 1.5))

def sample_indices(n_total, max_n):
    if n_total <= max_n:
        return np.arange(n_total)
    return np.random.choice(n_total, size=max_n, replace=False)

# =============================================================================
# OUTPUT
# =============================================================================

mtb.set_note(
    f"Hybrid Ensemble Results  |  Target: {target_col}  |  "
    f"Predictors: {n_features}  |  Rows: {n_rows}  |  Split: 70/30"
)

# -----------------------------------------------------------------------------
# SECTION 1 — Numeric Performance Metrics
# -----------------------------------------------------------------------------
mtb.add_table(
    columns=[
        ["Train R-squared (%)", "Validation R-squared (%)",
         "Train RMSE", "Validation RMSE", "Mean Absolute Deviation (MAD)"],
        [round(r2_train * 100, 4), round(r2_test * 100, 4),
         round(rmse_train, 4),     round(rmse_test, 4),
         round(mad, 4)],
    ],
    headers=["Metric", "Value"],
    title="Hybrid Ensemble — Performance Metrics",
    footnote=f"Target: {target_col}  |  Training rows: {len(y_train)}  |  Validation rows: {len(y_test)}"
)

# -----------------------------------------------------------------------------
# SECTION 2 — Ensemble Configuration & Weighting
# -----------------------------------------------------------------------------
mtb.add_table(
    columns=[
        ["Base 1: Deep ANN", "Base 2: Random Forest", "Base 1 (ANN) Ensemble Weight", "Base 2 (Tree) Ensemble Weight", "Meta-Learner"],
        [f"Neurons: {arch_str} | Epochs: {n_iter}",
         f"Trees: 100 | Max Depth: {rf_step.max_depth}",
         f"{ann_reliance:.1f} %", 
         f"{rf_reliance:.1f} %",
         "Ridge Regression (CV)"],
    ],
    headers=["Model Component", "Configuration / Influence"],
    title="Hybrid Ensemble — Architecture & Reliance",
    footnote="Ensemble weights show how heavily the meta-learner relied on each base model's predictions."
)

# -----------------------------------------------------------------------------
# SECTION 3 — Permutation Feature Importance Table
# -----------------------------------------------------------------------------
try:
    perm_result = permutation_importance(
        model, X_test_df, y_test,
        n_repeats    = PERM_IMP_REPEATS,
        random_state = 42,
        scoring      = "r2",
        n_jobs       = -1,
    )
    perm_mean = perm_result.importances_mean
    perm_std  = perm_result.importances_std

    sorted_idx = np.argsort(perm_mean)[::-1]
    top_n      = min(MAX_FEATURES_TABLE, n_features)

    mtb.add_table(
        columns=[
            list(range(1, top_n + 1)),
            [final_features[i] for i in sorted_idx[:top_n]],
            [round(float(perm_mean[i]), 6) for i in sorted_idx[:top_n]],
            [round(float(perm_std[i]),  6) for i in sorted_idx[:top_n]],
        ],
        headers=["Rank", "Feature", "Mean R² Drop", "Std Dev"],
        title=f"Hybrid Ensemble — Permutation Feature Importance (Top {top_n} of {n_features})",
        footnote=(
            f"Mean decrease in R² when each feature is randomly shuffled "
            f"({PERM_IMP_REPEATS} repeats).  Higher = stronger influence on predictions."
        )
    )
except Exception as e:
    perm_mean = None
    mtb.add_message(f"Note: Permutation importance skipped — {e}")

# -----------------------------------------------------------------------------
# PLOT 1 — Feature Importance Bar Chart
# -----------------------------------------------------------------------------
try:
    if perm_mean is not None:
        top_n_plot = min(MAX_FEATURES_PLOT, n_features)
        top_idx    = np.argsort(perm_mean)[::-1][:top_n_plot]
        feats_plot = [final_features[i] for i in reversed(top_idx)]
        vals_plot  = perm_mean[list(reversed(top_idx))]
        stds_plot  = perm_std[list(reversed(top_idx))]
        colors     = plt.cm.Greens(np.linspace(0.4, 0.9, top_n_plot))

        fig, ax = plt.subplots(figsize=(9, bar_chart_height(top_n_plot)))
        ax.barh(feats_plot, vals_plot, color=colors,
                xerr=stds_plot,
                error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"})
        ax.set_xlabel("Mean Decrease in R²  (higher = more important)")
        ax.set_title(
            f"Hybrid Ensemble — Permutation Feature Importance\n"
            f"Top {top_n_plot} of {n_features} Predictors",
            fontweight="bold"
        )
        ax.axvline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        save_and_push(fig, "Hybrid_feature_importance.png")
except Exception as e:
    mtb.add_message(f"Note: Feature importance chart skipped — {e}")

# -----------------------------------------------------------------------------
# PLOT 2 — Training Loss Curve (For the inner ANN base model)
# -----------------------------------------------------------------------------
try:
    loss_curve = list(getattr(ann_step, "loss_curve_", []))
    val_scores = list(getattr(ann_step, "validation_scores_", []))

    if loss_curve:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(loss_curve, color="mediumpurple", linewidth=1.8, label="Training Loss (MSE)")
        if val_scores:
            ax2 = ax.twinx()
            ax2.plot(val_scores, color="darkorange", linewidth=1.8,
                     linestyle="--", label="Internal Val R²")
            ax2.set_ylabel("Internal Validation R²", color="darkorange")
            ax2.tick_params(axis="y", colors="darkorange")
            ax2.legend(loc="center right", fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss (MSE)")
        ax.set_title(
            f"Hybrid Ensemble — Internal Base ANN Training Curve\n"
            f"Stopped at epoch {n_iter}  |  Early stopping patience = 40",
            fontweight="bold"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        save_and_push(fig, "Hybrid_ANN_loss_curve.png")
except Exception as e:
    mtb.add_message(f"Note: Loss curve skipped — {e}")

# -----------------------------------------------------------------------------
# PLOT 3 — Actual vs Predicted
# -----------------------------------------------------------------------------
idx_scatter = sample_indices(len(y_test), SCATTER_MAX_POINTS)
note_sub    = (f"  (sub-sampled {SCATTER_MAX_POINTS} of {len(y_test)} pts)"
               if len(y_test) > SCATTER_MAX_POINTS else "")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_true, y_pred, label, idx in [
    (axes[0], y_train, y_train_pred, "Training",
     sample_indices(len(y_train), SCATTER_MAX_POINTS)),
    (axes[1], y_test,  y_test_pred,  "Validation", idx_scatter),
]:
    mn = min(float(y_true[idx].min()), float(y_pred[idx].min()))
    mx = max(float(y_true[idx].max()), float(y_pred[idx].max()))
    ax.scatter(y_pred[idx], y_true[idx],
               color="seagreen", s=max(4, 40 - len(idx)//50), alpha=0.55)
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(label)
    ax.legend(fontsize=8)
fig.suptitle(f"Hybrid Ensemble — Actual vs Predicted{note_sub}", fontweight="bold")
fig.tight_layout()
save_and_push(fig, "Hybrid_actual_vs_predicted.png")

# -----------------------------------------------------------------------------
# PLOT 4 — Residual Analysis
# -----------------------------------------------------------------------------
res_idx  = idx_scatter
res_sub  = residuals[res_idx]
pred_sub = y_test_pred[res_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(pred_sub, res_sub,
                color="seagreen", s=max(4, 40 - len(res_idx)//50), alpha=0.55)
axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
axes[0].set_xlabel("Predicted Values")
axes[0].set_ylabel("Residuals  (Actual − Predicted)")
axes[0].set_title(f"Residuals vs Fitted{note_sub}")

n_bins = min(60, max(20, len(y_test) // 20))
axes[1].hist(residuals, bins=n_bins, color="seagreen", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
axes[1].set_xlabel("Residual Value")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution  (all validation rows)")

fig.suptitle("Hybrid Ensemble — Residual Analysis (Validation Set)", fontweight="bold")
fig.tight_layout()
save_and_push(fig, "Hybrid_residual_analysis.png")

# -----------------------------------------------------------------------------
# PLOTS 5–8 — SHAP  (KernelExplainer wrapper for Ensemble)
# -----------------------------------------------------------------------------
try:
    shap_idx    = sample_indices(len(X_test_df), SHAP_MAX_ROWS)
    X_shap      = X_test_df.iloc[shap_idx].reset_index(drop=True)
    y_shap_true = y_test[shap_idx]
    y_shap_pred = y_test_pred[shap_idx]
    shap_note   = (f"  (SHAP on {len(shap_idx)} of {len(X_test_df)} val rows)"
                   if len(X_test_df) > SHAP_MAX_ROWS else "")

    bg_idx      = sample_indices(len(X_train_df), SHAP_BACKGROUND_ROWS)
    X_bg        = shap.kmeans(X_train_df.iloc[bg_idx], min(40, SHAP_BACKGROUND_ROWS))

    explainer   = shap.KernelExplainer(model.predict, X_bg)
    shap_values = explainer.shap_values(X_shap, nsamples=80, silent=True)

    shap_display = min(SHAP_MAX_DISPLAY, n_features)

    # PLOT 5 — SHAP Beeswarm
    fig = plt.figure(figsize=(10, bar_chart_height(shap_display)))
    shap.summary_plot(shap_values, X_shap, plot_type="dot",
                      show=False, max_display=shap_display)
    plt.title(
        f"Hybrid Ensemble — SHAP Beeswarm{shap_note}\n"
        f"Top {shap_display} of {n_features} features",
        fontweight="bold", pad=12
    )
    fig = plt.gcf()
    fig.tight_layout()
    save_and_push(fig, "Hybrid_shap_beeswarm.png")

    # PLOT 6 — SHAP Waterfall (worst-predicted obs)
    worst_idx   = int(np.argmax(np.abs(y_shap_true - y_shap_pred)))
    explanation = shap.Explanation(
        values        = shap_values[worst_idx],
        base_values   = explainer.expected_value,
        data          = X_shap.iloc[worst_idx].values,
        feature_names = final_features,
    )
    fig = plt.figure(figsize=(10, bar_chart_height(min(shap_display, n_features))))
    shap.waterfall_plot(explanation, show=False, max_display=shap_display)
    plt.title(
        f"Hybrid Ensemble — SHAP Waterfall{shap_note}\n"
        f"Obs #{worst_idx+1}  "
        f"(Actual={y_shap_true[worst_idx]:.3f}, Predicted={y_shap_pred[worst_idx]:.3f})",
        fontweight="bold", pad=12
    )
    fig = plt.gcf()
    fig.tight_layout()
    save_and_push(fig, "Hybrid_shap_waterfall.png")

    # PLOT 7 — SHAP Dependence (top × second feature)
    mean_abs    = np.mean(np.abs(shap_values), axis=0)
    top2        = np.argsort(mean_abs)[::-1][:2]
    top_feat    = final_features[top2[0]]
    color_feat  = final_features[top2[1]]

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(top_feat, shap_values, X_shap,
                         interaction_index=color_feat, ax=ax, show=False)
    ax.set_title(
        f"Hybrid Ensemble — SHAP Dependence\n"
        f"'{top_feat}'  coloured by  '{color_feat}'{shap_note}",
        fontweight="bold"
    )
    fig.tight_layout()
    save_and_push(fig, "Hybrid_shap_dependence.png")

    # PLOT 8 — SHAP Force Summary Bar
    feat_contribs = pd.Series(shap_values[worst_idx], index=final_features)
    feat_contribs = feat_contribs.reindex(
        feat_contribs.abs().sort_values(ascending=True).index
    ).tail(shap_display)
    bar_colors = ["#d73027" if v > 0 else "#4575b4" for v in feat_contribs.values]

    fig, ax = plt.subplots(figsize=(10, bar_chart_height(min(shap_display, n_features))))
    feat_contribs.plot(kind="barh", color=bar_colors, ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value  (Impact on Prediction)")
    ax.set_title(
        f"Hybrid Ensemble — SHAP Force Summary\n"
        f"Obs #{worst_idx+1}  "
        f"(Base={explainer.expected_value:.3f}, "
        f"Predicted={y_shap_pred[worst_idx]:.3f}){shap_note}",
        fontweight="bold"
    )
    fig.tight_layout()
    save_and_push(fig, "Hybrid_shap_force_summary.png")

except Exception as e:
    mtb.add_message(f"Note: SHAP plots skipped — {e}")

gc.collect()

# -----------------------------------------------------------------------------
# PLOT 9 — Partial Dependence Plots
# -----------------------------------------------------------------------------
try:
    if perm_mean is not None:
        top_pdp_feats = [final_features[i]
                         for i in np.argsort(perm_mean)[::-1][:PDP_TOP_N_FEATURES]]
    else:
        top_pdp_feats = final_features[:PDP_TOP_N_FEATURES]

    n_pdp = len(top_pdp_feats)
    fig, axes = plt.subplots(1, n_pdp, figsize=(5 * n_pdp, 4))
    if n_pdp == 1:
        axes = [axes]

    PartialDependenceDisplay.from_estimator(
        model, X_train_df, features=top_pdp_feats,
        ax=axes, kind="average",
        line_kw={"color": "seagreen", "linewidth": 2}
    )
    for ax, feat in zip(axes, top_pdp_feats):
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel(feat)
        ax.set_ylabel("Partial Dependence")

    fig.suptitle(
        f"Hybrid Ensemble — Partial Dependence Plots (PDP)\n"
        f"Top {n_pdp} Features by Permutation Importance",
        fontweight="bold"
    )
    fig.tight_layout()
    save_and_push(fig, "Hybrid_pdp.png")
except Exception as e:
    mtb.add_message(f"Note: PDP skipped — {e}")

# -----------------------------------------------------------------------------
# PLOT 10 — ICE Plot (top feature)
# -----------------------------------------------------------------------------
try:
    top1_feat = (final_features[np.argmax(perm_mean)]
                 if perm_mean is not None else final_features[0])
    ice_lines = min(ICE_MAX_LINES, len(X_train_df))
    ice_alpha = max(0.05, min(0.3, 30 / ice_lines))

    fig, ax = plt.subplots(figsize=(8, 5))
    PartialDependenceDisplay.from_estimator(
        model, X_train_df, features=[top1_feat],
        ax=ax, kind="both", subsample=ice_lines,
        line_kw={"alpha": ice_alpha, "color": "seagreen"},
        pd_line_kw={"color": "red", "linewidth": 2.5, "label": "PDP mean"}
    )
    ax.set_title(
        f"Hybrid Ensemble — ICE Plot  (top feature: '{top1_feat}')\n"
        f"{ice_lines} individual lines  |  Red = PDP mean",
        fontweight="bold"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_and_push(fig, "Hybrid_ice_plot.png")
except Exception as e:
    mtb.add_message(f"Note: ICE plot skipped — {e}")

# -----------------------------------------------------------------------------
# PLOT 11 — Learning Curves
# -----------------------------------------------------------------------------
try:
    cv_folds = min(5, max(2, len(y_train) // 50))
    n_points = min(6, max(4, len(y_train) // 50))

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_df, y_train,
        cv          = cv_folds,
        train_sizes = np.linspace(0.2, 1.0, n_points),
        scoring     = "r2",
        n_jobs      = -1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="seagreen", label="Training R²")
    ax.plot(train_sizes, val_mean,   "o-", color="darkorange",   label="Cross-val R²")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="seagreen")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="darkorange")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R²")
    ax.set_title(
        f"Hybrid Ensemble — Learning Curves\n"
        f"Stability Across Training Sizes  ({cv_folds}-fold CV)",
        fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_and_push(fig, "Hybrid_learning_curves.png")
except Exception as e:
    mtb.add_message(f"Note: Learning curves skipped — {e}")

gc.collect()

# -----------------------------------------------------------------------------
# SECTION 4 — Predictions Table (Validation Set)
# -----------------------------------------------------------------------------
obs_indices = list(range(1, len(y_test) + 1))
actual_vals = [round(float(v), 4) for v in y_test]
pred_vals   = [round(float(v), 4) for v in y_test_pred]
abs_errors  = [round(abs(a - p), 4) for a, p in zip(actual_vals, pred_vals)]

mtb.add_table(
    columns=[obs_indices, actual_vals, pred_vals, abs_errors],
    headers=["Obs", "Actual", "Hybrid Predicted", "Absolute Error"],
    title="Hybrid Ensemble — Predictions (Validation Set)",
    footnote=(
        f"Validation set: {len(y_test)} observations (30 % holdout)  |  "
        f"Target: {target_col}  |  "
        f"Meta-Learner: RidgeCV"
    )
)