# =============================================================================
# Minitab Python Integration — Advanced Predictive Modelling Script
# Models : Decision Tree (DT), Random Forest (RF), XGBoost (XGB)
# Output : Model comparison, per-model metrics, feature importance,
#          SHAP plots, PDP, ICE, learning curves, residual diagnostics —
#          all displayed in the Minitab output pane
#
# GENERIC DESIGN — all limits auto-scale with your data:
#   • Features  : plot heights grow with feature count; top-N caps keep charts readable
#   • Rows      : SHAP/ICE/scatter use stratified sampling on large datasets
#   • Pred table: paginated into chunks of PRED_PAGE_SIZE rows so pane never overflows
#   • Tree depth: auto-capped based on feature count to keep schematics legible
#
# Dependencies (install once):
#   pip install scikit-learn xgboost shap matplotlib numpy pandas
#
# Arguments (passed from Minitab):
#   Training mode : <target_col> <max_predictor_col>
#                   e.g.  C5 C4   →  target=C5, predictors=C1–C4
#   Predict mode  : predict <target_col> <max_predictor_col> <model_name> <values>
#                   e.g.  predict C5 C4 rf 1.2,3.4,5.6,7.8
#                   model_name options: dt, rf, xgb
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORTANT — must be the very first two lines before ANY other import.
# Minitab runs Python in a background thread. The default matplotlib backend
# (TkAgg) requires tkinter which requires the main thread → crash.
# Setting MPLBACKEND via os.environ here blocks tkinter from loading at all,
# even if shap / lightgbm / xgboost try to trigger it internally.
# -----------------------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from mtbpy import mtbpy
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
import gc  # explicit garbage collection between models to free memory on large datasets

# =============================================================================
# GENERIC SCALING CONSTANTS
# Adjust these to tune behaviour for your dataset size without touching the
# rest of the code.
# =============================================================================

# Feature importance / coefficient tables & bar charts
MAX_FEATURES_PLOT   = 20    # max features shown in any bar chart
MAX_FEATURES_TABLE  = 25    # max rows in the feature importance table

# SHAP — sampling caps (TreeExplainer is fast but memory grows with n*features)
SHAP_MAX_ROWS       = 500   # max validation rows used for SHAP calculations
SHAP_MAX_DISPLAY    = 20    # max features in beeswarm / waterfall / force plots

# PDP / ICE
PDP_TOP_N_FEATURES  = 3     # number of features in the PDP panel (can increase)
ICE_MAX_LINES       = 150   # max individual lines drawn in the ICE plot

# Scatter / residual plots — sub-sample for readability on large validation sets
SCATTER_MAX_POINTS  = 1000  # max points in actual-vs-predicted and residual plots

# Tree schematics — max depth shown (deeper trees become unreadable)
TREE_PLOT_MAX_DEPTH = 4     # applies to RF representative tree and DT display

# =============================================================================

# -----------------------------------------------------------------------------
# 1. CONNECT TO MINITAB
# -----------------------------------------------------------------------------
mtb = mtbpy.mtb_instance()

# -----------------------------------------------------------------------------
# 2. SETUP — plots output folder (saved alongside this script)
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir  = os.path.join(script_dir, "Plots_Advanced")
os.makedirs(plots_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 3. PARSE ARGUMENTS passed in from Minitab
# -----------------------------------------------------------------------------
args         = sys.argv[1:]
predict_mode = args[0].lower() == "predict"

if predict_mode:
    _, target_col, max_col, model_name, input_str = args
    input_values = [float(x.strip()) for x in input_str.split(",")]
else:
    target_col, max_col = args

# -----------------------------------------------------------------------------
# 4. BUILD PREDICTOR COLUMN LIST  (C1 … max_col, excluding target)
# -----------------------------------------------------------------------------
max_idx        = int(max_col[1:])
predictor_cols = (
    [f"C{i}" for i in range(1, max_idx + 1) if f"C{i}" != target_col]
    if not predict_mode
    else [f"C{i}" for i in range(1, len(input_values) + 1)]
)

# -----------------------------------------------------------------------------
# 5. LOAD PREDICTOR DATA from Minitab worksheet (column by column)
# -----------------------------------------------------------------------------
X_data, valid_cols = [], []

for col in predictor_cols:
    try:
        X_data.append(mtb.get_column(col))
        valid_cols.append(col)
    except:
        mtb.add_message(f"Warning: Column {col} not found in worksheet — skipping.")

X_df           = pd.DataFrame(np.array(X_data).T, columns=valid_cols).dropna()
final_features = X_df.columns.tolist()

# -----------------------------------------------------------------------------
# 6. LOAD TARGET COLUMN from Minitab worksheet
# -----------------------------------------------------------------------------
try:
    y = np.array(mtb.get_column(target_col))[:len(X_df)]
except:
    mtb.add_message(f"Error: Target column '{target_col}' not found in worksheet.")
    sys.exit()

# -----------------------------------------------------------------------------
# 7. DEFINE MODELS
#    DT  — Decision Tree      : interpretable rules, prone to overfit
#    RF  — Random Forest      : robust ensemble, handles noise well
#    XGB — XGBoost            : high accuracy, built-in regularisation
# -----------------------------------------------------------------------------
MODEL_DISPLAY_NAMES = {
    "DT" : "Decision Tree (DT)",
    "RF" : "Random Forest (RF)",
    "XGB": "XGBoost (XGB)",
}

KEY_MAP = {
    "DT": "DT", "RF": "RF", "XGB": "XGB",
    "TREE": "DT", "FOREST": "RF", "XGBOOST": "XGB"
}

def build_models(n_features, n_rows):
    """
    Returns fresh untrained model instances with auto-scaled defaults.
    Hyperparameters auto-scale with dataset size (rows and feature count).
    """
    n_est      = min(400, max(100, int(n_rows / 10)))
    auto_depth = min(8, max(3, int(math.log2(n_features + 1)) + 2))

    return {
        "DT" : DecisionTreeRegressor(
                    max_depth        = min(auto_depth, TREE_PLOT_MAX_DEPTH + 1),
                    min_samples_leaf = max(3, int(n_rows * 0.01)),
                    random_state     = 42),
        "RF" : RandomForestRegressor(
                    n_estimators     = n_est,
                    max_depth        = auto_depth,
                    min_samples_leaf = max(3, int(n_rows * 0.005)),
                    random_state     = 42,
                    n_jobs           = -1),
        "XGB": xgb.XGBRegressor(
                    n_estimators     = n_est,
                    max_depth        = min(auto_depth, 8),
                    learning_rate    = 0.05,
                    subsample        = 0.8,
                    colsample_bytree = min(1.0, 20 / max(20, n_features)),
                    reg_alpha        = 0.1,
                    reg_lambda       = 1.0,
                    random_state     = 42,
                    verbosity        = 0),
    }



# Tree-based models do NOT need feature scaling —
# raw values give better SHAP interpretability

# -----------------------------------------------------------------------------
# 8. PREDICT MODE — train on full data, predict a single new observation
# -----------------------------------------------------------------------------
if predict_mode:
    model_name_key = KEY_MAP.get(model_name.upper())
    if model_name_key is None:
        mtb.add_message(
            f"Error: Model '{model_name}' not recognised. "
            f"Choose from: dt, rf, xgb."
        )
        sys.exit()

    models     = build_models(len(final_features), len(X_df))
    model      = models[model_name_key]
    model.fit(X_df, y)

    input_df   = pd.DataFrame([input_values], columns=final_features)
    prediction = model.predict(input_df)[0]

    mtb.add_message(
        f"Predicted Value : {prediction:.4f}\n"
        f"Model Used      : {MODEL_DISPLAY_NAMES[model_name_key]}\n"
        f"Target          : {target_col}"
    )
    sys.exit()

# -----------------------------------------------------------------------------
# 9. DATA CLEANING
#    • Remove near-zero variance columns  (uninformative predictors)
#    • Remove highly correlated columns   (>0.98 — improves SHAP stability)
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
    f"Dataset: {n_rows} rows  |  {n_features} predictors  |  Target: {target_col}"
)

# -----------------------------------------------------------------------------
# 10. TRAIN / VALIDATION SPLIT  (70% train — 30% validation)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=42
)

X_train_df = pd.DataFrame(X_train, columns=final_features)
X_test_df  = pd.DataFrame(X_test,  columns=final_features)

# -----------------------------------------------------------------------------
# 11. TRAIN ALL MODELS and collect performance metrics
# -----------------------------------------------------------------------------
models        = build_models(n_features, n_rows)
summary_list  = []
model_preds   = {}
model_details = {}

for name, model in models.items():

    model.fit(X_train_df, y_train)
    y_train_pred = model.predict(X_train_df)
    y_test_pred  = model.predict(X_test_df)

    r2_train   = r2_score(y_train, y_train_pred)
    r2_test    = r2_score(y_test,  y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
    mad        = np.mean(np.abs(y_test - y_test_pred))

    summary_list.append({
        "Model"     : name,
        "Val R2 (%)": round(r2_test * 100, 2),
        "MAD"       : round(mad, 4),
    })
    model_preds[name]   = model
    model_details[name] = {
        "r2_train"    : round(r2_train  * 100, 4),
        "r2_test"     : round(r2_test   * 100, 4),
        "rmse_train"  : round(rmse_train, 4),
        "rmse_test"   : round(rmse_test,  4),
        "mad"         : round(mad, 4),
        # Store as lists not numpy arrays — lower memory overhead
        "y_train_pred": y_train_pred.tolist(),
        "y_test_pred" : y_test_pred.tolist(),
    }

    # Free prediction arrays — they are now safely stored as lists above
    del y_train_pred, y_test_pred
    gc.collect()       # release prediction array memory before next model trains


best_idx = np.argmax([s["Val R2 (%)"] for s in summary_list])
for i, s in enumerate(summary_list):
    if i == best_idx:
        s["Model"] += " *"

summary_df    = pd.DataFrame(summary_list).sort_values("Val R2 (%)", ascending=False)
ordered_names = summary_df["Model"].str.replace(" *", "", regex=False).tolist()

# =============================================================================
# HELPERS
# =============================================================================

def save_and_push(fig, filename):
    """Save figure to Plots folder and push to Minitab output pane."""
    fpath = os.path.join(plots_dir, filename)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    mtb.add_image(fpath)
    plt.close(fig)


def bar_chart_height(n):
    """
    Compute a sensible figure height for horizontal bar charts.
    Scales linearly with bar count so labels never overlap.
    """
    return max(4, min(20, n * 0.45 + 1.5))


def sample_indices(n_total, max_n):
    """
    Return a stratified random index array of size min(n_total, max_n).
    Used to sub-sample large validation sets for scatter / SHAP plots.
    """
    if n_total <= max_n:
        return np.arange(n_total)
    return np.random.choice(n_total, size=max_n, replace=False)


# =============================================================================
# OUTPUT SECTION
# =============================================================================

mtb.set_note(
    f"Advanced Predictive Modelling Results  |  "
    f"Target: {target_col}  |  "
    f"Predictors: {n_features}  |  "
    f"Rows: {n_rows}  |  "
    f"Train / Validation split: 70 / 30"
)

# -----------------------------------------------------------------------------
# SECTION 1 — Overall Model Comparison
# -----------------------------------------------------------------------------
mtb.add_table(
    columns=[
        summary_df["Model"].tolist(),
        summary_df["Val R2 (%)"].tolist(),
        summary_df["MAD"].tolist(),
    ],
    headers=["Model", "Validation R-squared (%)", "Mean Absolute Deviation"],
    title="Overall Model Comparison",
    footnote="* Best model by Validation R-squared"
)

# -----------------------------------------------------------------------------
# SECTION 2 — Individual Model Results  (best → worst)
#
#   Per model:
#     2a.  Performance metrics table
#     2b.  Feature importance table   (top N, capped at MAX_FEATURES_TABLE)
#     2c.  Feature Importance Bar Chart
#     2d.  Actual vs Predicted        (sub-sampled if > SCATTER_MAX_POINTS)
#     2e.  Residuals Plot             (sub-sampled if > SCATTER_MAX_POINTS)
#     2f.  SHAP Beeswarm              (sampled to SHAP_MAX_ROWS)
#     2g.  SHAP Waterfall             (worst-predicted observation)
#     2h.  SHAP Dependence Plot       (top feature × second feature)
#     2i.  SHAP Force Summary Bar     (worst-predicted observation)
#     2j.  PDP                        (top PDP_TOP_N_FEATURES features)
#     2k.  ICE Plot                   (capped at ICE_MAX_LINES)
#     2l.  Tree Schematic             (depth capped at TREE_PLOT_MAX_DEPTH)
#     2m.  Learning Curves
# -----------------------------------------------------------------------------

for name in ordered_names:

    model        = model_preds[name]
    details      = model_details[name]
    display_name = MODEL_DISPLAY_NAMES[name]


    # Convert stored lists back to numpy arrays for plot/metric calculations
    y_train_pred = np.array(details["y_train_pred"])
    y_test_pred  = np.array(details["y_test_pred"])
    residuals    = y_test - y_test_pred

    # -- 2a. Performance metrics table --------------------------------------
    metric_labels = [
        "Train R-squared (%)",
        "Validation R-squared (%)",
        "Train RMSE",
        "Validation RMSE",
        "Mean Absolute Deviation (MAD)",
    ]
    metric_values = [
        details["r2_train"], details["r2_test"],
        details["rmse_train"], details["rmse_test"],
        details["mad"],
    ]

    if name == "DT":
        metric_labels += ["Tree Depth (actual)", "Number of Leaves"]
        metric_values += [model.get_depth(), model.get_n_leaves()]
    elif name == "RF":
        metric_labels += ["N Estimators", "Max Depth"]
        metric_values += [model.n_estimators, model.max_depth]
    elif name == "XGB":
        metric_labels += ["N Estimators", "Learning Rate", "Max Depth"]
        metric_values += [model.n_estimators, model.learning_rate, model.max_depth]

    mtb.add_table(
        columns=[metric_labels, metric_values],
        headers=["Metric", "Value"],
        title=f"{display_name} — Performance Metrics",
        footnote=(
            f"Target: {target_col}  |  Predictors: {n_features}  |  "
            f"Training rows: {len(y_train)}  |  Validation rows: {len(y_test)}"
        )
    )

    # -- 2b. Feature importance table (top N, scaled to data) ---------------
    if hasattr(model, "feature_importances_"):
        fi        = model.feature_importances_
        fi_sorted = np.argsort(fi)[::-1]
        top_n_tbl = min(MAX_FEATURES_TABLE, n_features)
        top_feats = [final_features[i] for i in fi_sorted[:top_n_tbl]]
        top_imps  = [round(float(fi[i]), 6) for i in fi_sorted[:top_n_tbl]]
        top_ranks = list(range(1, top_n_tbl + 1))

        mtb.add_table(
            columns=[top_ranks, top_feats, top_imps],
            headers=["Rank", "Feature", "Importance Score"],
            title=f"{display_name} — Feature Importance (Top {top_n_tbl} of {n_features})",
            footnote=(
                "Importance = mean decrease in impurity (MDI) across all trees.  "
                "Higher score = stronger influence on predictions."
            )
        )

    # ==========================================================================
    # 2c. Feature Importance Bar Chart
    #     Height scales with number of features shown so labels never overlap
    # ==========================================================================
    if hasattr(model, "feature_importances_"):
        fi      = model.feature_importances_
        abs_idx = np.argsort(fi)[::-1]
        top_n   = min(MAX_FEATURES_PLOT, n_features)
        top_idx = abs_idx[:top_n]

        fig_h  = bar_chart_height(top_n)
        colors = plt.cm.Blues(np.linspace(0.35, 0.9, top_n))[::-1]

        fig, ax = plt.subplots(figsize=(9, fig_h))
        ax.barh(
            [final_features[i] for i in reversed(top_idx)],
            fi[list(reversed(top_idx))],
            color=colors
        )
        ax.set_xlabel("Importance Score (Mean Decrease in Impurity)")
        ax.set_title(
            f"{display_name}\nFeature Importance — Top {top_n} of {n_features} Predictors",
            fontweight="bold"
        )
        ax.axvline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        save_and_push(fig, f"{name}_feature_importance.png")

    # ==========================================================================
    # 2d. Actual vs Predicted — sub-sampled if validation set is large
    # ==========================================================================
    idx_scatter = sample_indices(len(y_test), SCATTER_MAX_POINTS)
    note_sub    = (f"  (sub-sampled {SCATTER_MAX_POINTS} of {len(y_test)} pts)"
                   if len(y_test) > SCATTER_MAX_POINTS else "")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_true, y_pred, split_label, idx in [
        (axes[0], y_train, y_train_pred,
         "Training",   sample_indices(len(y_train), SCATTER_MAX_POINTS)),
        (axes[1], y_test,  y_test_pred,
         "Validation", idx_scatter),
    ]:
        mn = min(float(y_true[idx].min()), float(y_pred[idx].min()))
        mx = max(float(y_true[idx].max()), float(y_pred[idx].max()))
        ax.scatter(y_pred[idx], y_true[idx],
                   color="steelblue", s=max(4, 40 - len(idx) // 50), alpha=0.55)
        ax.plot([mn, mx], [mn, mx], color="black", linewidth=1,
                linestyle="--", label="Perfect fit")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(split_label)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"{display_name} — Actual vs Predicted{note_sub}",
        fontweight="bold"
    )
    fig.tight_layout()
    save_and_push(fig, f"{name}_actual_vs_predicted.png")

    # ==========================================================================
    # 2e. Residuals Plot — sub-sampled if large
    # ==========================================================================
    res_idx  = idx_scatter
    res_sub  = residuals[res_idx]
    pred_sub = y_test_pred[res_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(pred_sub, res_sub,
                    color="steelblue", s=max(4, 40 - len(res_idx) // 50), alpha=0.55)
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals  (Actual − Predicted)")
    axes[0].set_title(f"Residuals vs Fitted{note_sub}")

    n_bins = min(60, max(20, len(y_test) // 20))
    axes[1].hist(residuals, bins=n_bins, color="steelblue",
                 edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution  (all validation rows)")

    fig.suptitle(
        f"{display_name} — Residual Analysis (Validation Set)",
        fontweight="bold"
    )
    fig.tight_layout()
    save_and_push(fig, f"{name}_residual_analysis.png")

    # ==========================================================================
    # 2f–2i. SHAP Plots
    #        All SHAP calculations use a stratified sample of the validation set
    #        (capped at SHAP_MAX_ROWS) so memory and time stay bounded.
    # ==========================================================================
    try:
        # Sub-sample validation set for SHAP if needed
        shap_idx    = sample_indices(len(X_test_df), SHAP_MAX_ROWS)
        X_shap      = X_test_df.iloc[shap_idx].reset_index(drop=True)
        y_shap_true = y_test[shap_idx]
        y_shap_pred = y_test_pred[shap_idx]
        shap_note   = (f"  (SHAP computed on {len(shap_idx)} of {len(X_test_df)} "
                       f"validation rows)" if len(X_test_df) > SHAP_MAX_ROWS else "")

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        # -- 2f. SHAP Beeswarm / Summary Plot ---------------------------------
        #    Dot height = SHAP_MAX_DISPLAY features; plot height scales with it
        shap_display = min(SHAP_MAX_DISPLAY, n_features)
        fig_h_shap   = bar_chart_height(shap_display)

        fig = plt.figure(figsize=(10, fig_h_shap))
        shap.summary_plot(
            shap_values, X_shap,
            plot_type="dot",
            show=False,
            max_display=shap_display
        )
        plt.title(
            f"{display_name} — SHAP Summary (Beeswarm){shap_note}\n"
            f"Feature Impact on Predictions  (top {shap_display} of {n_features})",
            fontweight="bold", pad=12
        )
        fig = plt.gcf()
        fig.tight_layout()
        save_and_push(fig, f"{name}_shap_summary_beeswarm.png")

        # -- 2g. SHAP Waterfall — worst-predicted observation in SHAP sample --
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
            f"{display_name} — SHAP Waterfall{shap_note}\n"
            f"Explaining Observation #{worst_idx + 1}  "
            f"(Actual={y_shap_true[worst_idx]:.3f}, "
            f"Predicted={y_shap_pred[worst_idx]:.3f})",
            fontweight="bold", pad=12
        )
        fig = plt.gcf()
        fig.tight_layout()
        save_and_push(fig, f"{name}_shap_waterfall.png")

        # -- 2h. SHAP Dependence Plot — top × second feature interaction ------
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top2_idx      = np.argsort(mean_abs_shap)[::-1][:2]
        top_feat      = final_features[top2_idx[0]]
        interact_feat = final_features[top2_idx[1]]

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            top_feat, shap_values, X_shap,
            interaction_index=interact_feat,
            ax=ax, show=False
        )
        ax.set_title(
            f"{display_name} — SHAP Dependence Plot\n"
            f"'{top_feat}'  coloured by  '{interact_feat}'{shap_note}",
            fontweight="bold"
        )
        fig.tight_layout()
        save_and_push(fig, f"{name}_shap_dependence.png")

        # -- 2i. SHAP Force Summary Bar — signed contributions ----------------
        fig_h_force = bar_chart_height(min(shap_display, n_features))
        feat_contributions = pd.Series(
            shap_values[worst_idx], index=final_features
        ).reindex(
            pd.Series(shap_values[worst_idx], index=final_features)
            .abs().sort_values(ascending=True).index
        ).tail(shap_display)

        colors = ["#d73027" if v > 0 else "#4575b4"
                  for v in feat_contributions.values]

        fig, ax = plt.subplots(figsize=(10, fig_h_force))
        feat_contributions.plot(kind="barh", color=colors, ax=ax)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value  (Impact on Prediction)")
        ax.set_title(
            f"{display_name} — SHAP Force Summary\n"
            f"Observation #{worst_idx + 1}  "
            f"(Base={explainer.expected_value:.3f}, "
            f"Predicted={y_shap_pred[worst_idx]:.3f}){shap_note}",
            fontweight="bold"
        )
        fig.tight_layout()
        save_and_push(fig, f"{name}_shap_force_summary.png")

    except Exception as e:
        mtb.add_message(f"Note: SHAP plots skipped for {display_name} — {e}")

    # ==========================================================================
    # 2j. Partial Dependence Plots (PDP)
    #     Top PDP_TOP_N_FEATURES features, each in its own subplot.
    #     Panel layout dynamically adjusts to feature count.
    # ==========================================================================
    try:
        if hasattr(model, "feature_importances_"):
            top_pdp_idx   = np.argsort(model.feature_importances_)[::-1][:PDP_TOP_N_FEATURES]
            top_pdp_feats = [final_features[i] for i in top_pdp_idx]
        else:
            top_pdp_feats = final_features[:PDP_TOP_N_FEATURES]

        n_pdp  = len(top_pdp_feats)
        fig, axes = plt.subplots(1, n_pdp, figsize=(5 * n_pdp, 4))
        if n_pdp == 1:
            axes = [axes]

        PartialDependenceDisplay.from_estimator(
            model, X_train_df,
            features  = top_pdp_feats,
            ax        = axes,
            kind      = "average",
            line_kw   = {"color": "steelblue", "linewidth": 2}
        )
        for ax, feat in zip(axes, top_pdp_feats):
            ax.set_title(feat, fontsize=10)
            ax.set_xlabel(feat)
            ax.set_ylabel("Partial Dependence")

        fig.suptitle(
            f"{display_name} — Partial Dependence Plots (PDP)\n"
            f"Top {n_pdp} Features  (of {n_features} total)",
            fontweight="bold"
        )
        fig.tight_layout()
        save_and_push(fig, f"{name}_pdp_top{n_pdp}.png")

    except Exception as e:
        mtb.add_message(f"Note: PDP skipped for {display_name} — {e}")

    # ==========================================================================
    # 2k. ICE Plot — individual responses for top feature
    #     Line count capped at ICE_MAX_LINES for readability
    # ==========================================================================
    try:
        top1_feat = (final_features[np.argmax(model.feature_importances_)]
                     if hasattr(model, "feature_importances_")
                     else final_features[0])

        ice_lines = min(ICE_MAX_LINES, len(X_train_df))
        ice_alpha = max(0.05, min(0.3, 30 / ice_lines))   # alpha fades as lines increase

        fig, ax = plt.subplots(figsize=(8, 5))
        PartialDependenceDisplay.from_estimator(
            model, X_train_df,
            features    = [top1_feat],
            ax          = ax,
            kind        = "both",
            subsample   = ice_lines,
            line_kw     = {"alpha": ice_alpha, "color": "steelblue"},
            pd_line_kw  = {"color": "red", "linewidth": 2.5, "label": "PDP mean"}
        )
        ax.set_title(
            f"{display_name} — ICE Plot  (top feature: '{top1_feat}')\n"
            f"{ice_lines} individual lines shown  |  Red = PDP mean",
            fontweight="bold"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_and_push(fig, f"{name}_ice_plot.png")

    except Exception as e:
        mtb.add_message(f"Note: ICE plot skipped for {display_name} — {e}")

    # ==========================================================================
    # 2l. Tree Schematic — rules of thumb for operators
    #     Depth is capped at TREE_PLOT_MAX_DEPTH for all models.
    #     Figure width scales with 2^depth so nodes stay readable.
    # ==========================================================================
    try:
        tree_depth = TREE_PLOT_MAX_DEPTH
        fig_w      = min(24, max(14, 2 ** tree_depth * 1.2))
        fig_h_tree = max(6, tree_depth * 2.5)

        if name == "DT":
            fig, ax = plt.subplots(figsize=(fig_w, fig_h_tree))
            plot_tree(
                model, feature_names=final_features,
                filled=True, rounded=True,
                fontsize=max(5, min(9, 60 // n_features)),
                max_depth=tree_depth,
                ax=ax, impurity=False, precision=3
            )
            ax.set_title(
                f"{display_name} — Decision Tree Schematic\n"
                f"(Displayed depth: {tree_depth}  |  "
                f"Actual depth: {model.get_depth()})",
                fontweight="bold"
            )
            fig.tight_layout()
            save_and_push(fig, f"{name}_tree_schematic.png")

        elif name == "RF":
            single_tree = model.estimators_[0]
            fig, ax     = plt.subplots(figsize=(fig_w, fig_h_tree))
            plot_tree(
                single_tree, feature_names=final_features,
                filled=True, rounded=True,
                fontsize=max(5, min(9, 60 // n_features)),
                max_depth=tree_depth,
                ax=ax, impurity=False, precision=3
            )
            ax.set_title(
                f"{display_name} — Representative Tree #1\n"
                f"(Displayed depth: {tree_depth}  |  "
                f"Forest size: {model.n_estimators} trees)",
                fontweight="bold"
            )
            fig.tight_layout()
            save_and_push(fig, f"{name}_tree_schematic.png")

        elif name == "XGB":
            fig, ax = plt.subplots(figsize=(fig_w, fig_h_tree))
            xgb.plot_tree(model, num_trees=0, ax=ax)
            ax.set_title(
                f"{display_name} — First Booster Tree (#0)\n"
                f"(Ensemble size: {model.n_estimators} trees)",
                fontweight="bold"
            )
            fig.tight_layout()
            save_and_push(fig, f"{name}_tree_schematic.png")

    except Exception as e:
        mtb.add_message(f"Note: Tree schematic skipped for {display_name} — {e}")

    # ==========================================================================
    # 2m. Learning Curves — model stability across training sizes
    #     CV folds auto-capped to 5 (or fewer if training set is small).
    # ==========================================================================
    try:
        cv_folds = min(5, max(2, len(y_train) // 50))
        n_points = min(10, max(5, len(y_train) // 50))

        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train_df, y_train,
            cv          = cv_folds,
            train_sizes = np.linspace(0.1, 1.0, n_points),
            scoring     = "r2",
            n_jobs      = -1
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_mean, "o-", color="steelblue",  label="Training R²")
        ax.plot(train_sizes, val_mean,   "o-", color="darkorange", label="Cross-val R²")
        ax.fill_between(train_sizes,
                        train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="steelblue")
        ax.fill_between(train_sizes,
                        val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="darkorange")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("R²  Score")
        ax.set_title(
            f"{display_name} — Learning Curves\n"
            f"Model Stability Across Training Sizes  ({cv_folds}-fold CV)",
            fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_and_push(fig, f"{name}_learning_curves.png")

    except Exception as e:
        mtb.add_message(f"Note: Learning curves skipped for {display_name} — {e}")

    # -------------------------------------------------------------------------
    # END OF MODEL LOOP — memory cleanup
    # On large datasets each model can consume several GB of RAM.
    # Explicitly deleting arrays and calling gc.collect() forces Python to
    # release that memory before the next model's plots are generated,
    # preventing the silent OOM failure where subsequent models produce nothing.
    # -------------------------------------------------------------------------
    del y_train_pred, y_test_pred, residuals
    plt.close("all")   # close any figures left open by SHAP / PDP / ICE
    gc.collect()       # force memory release back to OS


# -----------------------------------------------------------------------------
# SECTION 3 — Predictions Table  (Validation set, single table — all rows)
# -----------------------------------------------------------------------------
obs_indices = list(range(1, len(y_test) + 1))
actual_vals = [round(float(v), 4) for v in y_test]
pred_dt     = [round(float(v), 4) for v in model_preds["DT"].predict(X_test_df)]
pred_rf     = [round(float(v), 4) for v in model_preds["RF"].predict(X_test_df)]
pred_xgb    = [round(float(v), 4) for v in model_preds["XGB"].predict(X_test_df)]

n_val = len(y_test)

mtb.add_table(
    columns=[obs_indices, actual_vals, pred_dt, pred_rf, pred_xgb],
    headers=["Obs", "Actual", "DT Pred", "RF Pred", "XGB Pred"],
    title="Predictions — All Models (Validation Set)",
    footnote=(
        f"Validation set: {n_val} observations (30% holdout)  |  "
        f"Target: {target_col}  |  "
        f"Residual diagnostics available in the plots above"
    )
)