# =============================================================================
# Minitab Python Integration — Autoencoder Anomaly Detection
# Process    : OSB / Wood Panel Manufacturing
# Framework  : scikit-learn + numpy ONLY
#              NO PyTorch, NO TensorFlow, NO DLL issues
#              Works on any Python version including 3.13
#
# Install (one-time):
#   pip install scikit-learn matplotlib numpy pandas
#
# Model  : Sliding-window Autoencoder using stacked MLPRegressor
#          Trained on normal windows -> high reconstruction error = anomaly
#
# =============================================================================
# DATA FORMAT  (OSB Manufacturing — 15 process tags + optional label)
# =============================================================================
#
#   One row per hourly timestamp from the production line.
#
#   C1-T  timestamp          — Text column (YYYY-MM-DD HH:MM:SS)
#   C2    log_moisture       — Raw log moisture content (%)
#   C3    wood_density       — Strand density (kg/m3)
#   C4    strand_length      — Strand length (mm)
#   C5    dryer_temp         — Rotary dryer temperature (deg C)
#   C6    dryer_humidity     — Dryer exhaust humidity (%)
#   C7    drying_time        — Strand drying time (s)
#   C8    resin_ratio        — Resin application ratio (%)
#   C9    wax_content        — Wax content (%)
#   C10   blending_speed     — Blender drum speed (RPM)
#   C11   press_temp         — Hot press temperature (deg C)
#   C12   press_pressure     — Press platen pressure (MPa)
#   C13   press_time         — Press cycle time (s)
#   C14   motor_vibration    — Drive motor vibration
#   C15   energy_consumption — Line energy consumption (kWh)
#   C16   roller_speed       — Exit roller speed (m/min)
#   C17   anomaly            — OPTIONAL label (0=normal, 1=anomaly)
#
#   Process sections:
#     Forming  : C2  log_moisture, C3 wood_density, C4 strand_length
#     Blending : C8  resin_ratio,  C9 wax_content,  C10 blending_speed
#     Dryer    : C5  dryer_temp,   C6 dryer_humidity, C7 drying_time
#     Press    : C11 press_temp,   C12 press_pressure, C13 press_time
#     Mechanical: C14 motor_vibration, C15 energy_consumption, C16 roller_speed
#
# =============================================================================
# MINITAB ARGUMENTS
# =============================================================================
#
#   Without anomaly labels:   C1 C16
#   With anomaly labels:      C1 C16 C17
#
# =============================================================================

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mtbpy import mtbpy
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix, roc_curve,
                              precision_recall_curve)

warnings.filterwarnings("ignore")
import gc

# =============================================================================
# TUNING CONSTANTS
# =============================================================================

# Windowing
# OSB press cycles are typically 3–5 min; with hourly data a 12-hour window
# captures one shift. Use 24 for a full day (recommended starting point).
WINDOW_SIZE      = 24    # rows per window (24h for hourly data = 1 shift pair)
WINDOW_STRIDE    =  6    # step between windows (6h = 75% overlap)

# Model architecture
BOTTLENECK_RATIO = 0.25  # bottleneck compression — lower = more sensitive

# Training
MAX_ITER         = 500   # max iterations (early stopping fires before this)
LEARNING_RATE    = 1e-3  # Adam learning rate
ES_PATIENCE      = 20    # early stopping patience
VAL_SPLIT        = 0.1   # validation fraction

# Anomaly threshold
THRESHOLD_K      = 3.0   # mean + K*std of training MAE
                         # increase K -> fewer alarms; decrease -> more sensitive

# Plot limits
MAX_RECON_PLOTS  =  6    # sample reconstruction panels

# =============================================================================
# ENHANCEMENT CONSTANTS  (OSB Manufacturing)
# =============================================================================

# Alarm severity tiers
SEVERITY_WARNING_MULT  = 1.20   # 20% above threshold -> Warning
SEVERITY_CRITICAL_MULT = 1.50   # 50% above threshold -> Critical

# Consecutive window filter
# OSB lines have short transient upsets during press loading/unloading.
# Require 2 consecutive windows to avoid flagging those transients.
MIN_CONSECUTIVE_WINDOWS = 2

# Suppression feature
# No binary on/off flag in this dataset — set to None.
# If you later add a "line_stopped" or "grade_change" column, put its name here.
SUPPRESSION_FEATURE     = None

# Rolling health trend window (120 windows x 6h stride = 30 days)
ROLLING_HEALTH_WINDOWS  = 120

# =============================================================================
# PROCESS SECTION MAP
# Maps each feature name to its process section for richer alarm context.
# Used in the enhanced alarm log to show which section of the line is affected.
# =============================================================================
PROCESS_SECTION_MAP = {
    "log_moisture"      : "Forming",
    "wood_density"      : "Forming",
    "strand_length"     : "Forming",
    "dryer_temp"        : "Dryer",
    "dryer_humidity"    : "Dryer",
    "drying_time"       : "Dryer",
    "resin_ratio"       : "Blending",
    "wax_content"       : "Blending",
    "blending_speed"    : "Blending",
    "press_temp"        : "Press",
    "press_pressure"    : "Press",
    "press_time"        : "Press",
    "motor_vibration"   : "Mechanical",
    "energy_consumption": "Mechanical",
    "roller_speed"      : "Mechanical",
}

# =============================================================================

# ---- 1. CONNECT TO MINITAB --------------------------------------------------
mtb = mtbpy.mtb_instance()

# ---- 2. PLOTS FOLDER --------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir  = os.path.join(script_dir, "Plots_AE")
os.makedirs(plots_dir, exist_ok=True)

# ---- 3. PARSE ARGUMENTS -----------------------------------------------------
args         = sys.argv[1:]
ts_col       = args[0]
max_feat_col = args[1]
label_col    = args[2] if len(args) > 2 else None
has_labels   = label_col is not None

# ---- 4. LOAD DATA -----------------------------------------------------------
def load_col(col, text=False):
    try:
        if text:
            return mtb.get_column_as_text(col)
        return mtb.get_column(col)
    except Exception:
        try:
            return mtb.get_column(col)
        except Exception:
            mtb.add_message(f"Error: Column {col} not found.")
            sys.exit()

# ---- TIMESTAMP LOADING — robust multi-method fallback ----------------------
# Minitab stores date/time columns differently depending on column type:
#   C-T  (text)    → get_column_as_text() returns strings directly
#   C    (numeric) → get_column() returns floats (Minitab internal date serial)
#   Either can fail depending on Minitab version — so we try all methods.

timestamps = None
ts_source  = ""

# Method 1: text column (C-T type)
try:
    raw = mtb.get_column_as_text(ts_col)
    parsed = pd.to_datetime(raw, infer_datetime_format=True, errors="coerce")
    if parsed.notna().sum() > len(parsed) * 0.5:   # at least 50% parsed OK
        timestamps = parsed
        ts_source  = "text column (C-T)"
except Exception:
    pass

# Method 2: numeric column → try Minitab date serial (days since 1900-01-01)
if timestamps is None:
    try:
        raw_num = np.array(mtb.get_column(ts_col), dtype=float)
        # Minitab date serial: days since 1899-12-30 (same as Excel)
        parsed = pd.to_datetime(raw_num, unit="D", origin="1899-12-30", errors="coerce")
        if parsed.notna().sum() > len(parsed) * 0.5:
            timestamps = parsed
            ts_source  = "numeric date serial (Minitab/Excel origin)"
    except Exception:
        pass

# Method 3: numeric column → try Unix seconds
if timestamps is None:
    try:
        raw_num = np.array(mtb.get_column(ts_col), dtype=float)
        parsed  = pd.to_datetime(raw_num, unit="s", errors="coerce")
        if parsed.notna().sum() > len(parsed) * 0.5:
            timestamps = parsed
            ts_source  = "Unix timestamp (seconds)"
    except Exception:
        pass

# Method 4: fall back to plain text re-parse with dayfirst guessing
if timestamps is None:
    try:
        raw = mtb.get_column_as_text(ts_col)
        parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        if parsed.notna().sum() > len(parsed) * 0.5:
            timestamps = parsed
            ts_source  = "text column (dayfirst)"
    except Exception:
        pass

# Final fallback: integer row index — plots will show row numbers on x-axis
if timestamps is None or (hasattr(timestamps, "notna") and timestamps.isna().all()):
    n_rows_approx = len(feat_data[list(feat_data.keys())[0]]) if feat_data else 0
    timestamps = pd.RangeIndex(n_rows_approx)
    ts_source  = "row index (timestamp parsing failed)"
    mtb.add_message(
        f"Note: Could not parse column {ts_col} as timestamps.\n"
        f"Plots will use row index on x-axis instead of dates.\n"
        f"Tip: Make sure your timestamp column is set as C-T (text) type in Minitab.\n"
        f"     Right-click the column header > Column > select 'Text'."
    )
else:
    mtb.add_message(f"Timestamps loaded via: {ts_source}")

ts_idx       = int(ts_col[1:])
max_feat_idx = int(max_feat_col[1:])
feat_cols    = [f"C{i}" for i in range(ts_idx + 1, max_feat_idx + 1)]

feat_data = {}
for col in feat_cols:
    try:
        feat_data[col] = np.array(load_col(col), dtype=float)
    except Exception:
        mtb.add_message(f"Warning: Column {col} skipped.")

feature_names = list(feat_data.keys())
n_features    = len(feature_names)

if n_features == 0:
    mtb.add_message("Error: No feature columns found. Check arguments.")
    sys.exit()

df = pd.DataFrame(feat_data, columns=feature_names)
df.insert(0, "timestamp", timestamps)

if has_labels:
    df["anomaly"] = np.array(load_col(label_col), dtype=int)
    n_anom_rows   = int((df["anomaly"] == 1).sum())
else:
    df["anomaly"] = -1
    n_anom_rows   = 0

n_rows = len(df)
df[feature_names] = df[feature_names].ffill().bfill()

mtb.add_message(
    f"Data loaded : {n_rows} rows  |  {n_features} features  |  "
    f"Labels: {'Yes (' + str(n_anom_rows) + ' anomaly rows)' if has_labels else 'No (unsupervised)'}\n"
    f"Window: {WINDOW_SIZE} rows  |  Stride: {WINDOW_STRIDE}  |  "
    f"Approx windows: {max(0,(n_rows-WINDOW_SIZE)//WINDOW_STRIDE+1)}\n"
    f"Framework: scikit-learn MLPRegressor (no PyTorch/TensorFlow required)"
)

# ---- 5. SLIDING WINDOWS -----------------------------------------------------
def make_windows(arr2d, size, stride):
    """Slice continuous 2D array into overlapping 2D windows, flatten each."""
    starts  = range(0, len(arr2d) - size + 1, stride)
    # Flatten window to 1D: each window = size * n_features inputs
    windows = np.array([arr2d[s:s + size].flatten() for s in starts],
                       dtype=np.float32)
    return windows, list(starts)

raw_values = df[feature_names].values   # [n_rows, n_features]

# ---- 6. SCALE ---------------------------------------------------------------
if has_labels and (df["anomaly"] == 0).sum() > 0:
    normal_mask_rows = df["anomaly"].values == 0
else:
    normal_mask_rows = np.ones(n_rows, dtype=bool)

scaler = StandardScaler()
scaler.fit(raw_values[normal_mask_rows])
scaled = scaler.transform(raw_values).astype(np.float32)

# ---- 7. BUILD WINDOWS -------------------------------------------------------
all_windows, window_starts = make_windows(scaled, WINDOW_SIZE, WINDOW_STRIDE)
n_windows  = len(all_windows)
input_dim  = all_windows.shape[1]   # = WINDOW_SIZE * n_features

if has_labels:
    lbl_arr       = df["anomaly"].values
    window_labels = np.array([
        int(lbl_arr[s:s + WINDOW_SIZE].max())
        for s in window_starts
    ], dtype=int)
else:
    window_labels = np.full(n_windows, -1, dtype=int)

train_mask = (window_labels == 0) if has_labels else np.ones(n_windows, dtype=bool)
X_train    = all_windows[train_mask]

# Validation split
n_val  = max(1, int(len(X_train) * VAL_SPLIT))
X_val  = X_train[-n_val:]
X_tr   = X_train[:-n_val]

mtb.add_message(
    f"Windows: {n_windows} total  |  {len(X_tr)} train  |  "
    f"{len(X_val)} val  |  Input dim per window: {input_dim}"
)

# ---- 8. BUILD AUTOENCODER ---------------------------------------------------
#
#  Architecture (pyramid encoder → bottleneck → mirrored decoder):
#
#  Input (input_dim)
#    → Encoder layer 1 : input_dim * 0.75
#    → Encoder layer 2 : input_dim * 0.5
#    → Bottleneck       : input_dim * BOTTLENECK_RATIO
#    → Decoder layer 1 : input_dim * 0.5
#    → Decoder layer 2 : input_dim * 0.75
#    → Output           : input_dim  (reconstruct original window)
#
#  MLPRegressor treats reconstruction as a regression problem.
#  Loss = MSE between input and reconstructed output.
#  Anomaly score = MAE between input and reconstruction per window.

enc1 = max(32, int(input_dim * 0.75))
enc2 = max(16, int(input_dim * 0.50))
bnk  = max(8,  int(input_dim * BOTTLENECK_RATIO))
dec1 = enc2
dec2 = enc1

hidden_layers = (enc1, enc2, bnk, dec1, dec2)

autoencoder = MLPRegressor(
    hidden_layer_sizes  = hidden_layers,
    activation          = "relu",
    solver              = "adam",
    alpha               = 1e-4,           # mild L2 regularisation
    learning_rate       = "adaptive",
    learning_rate_init  = LEARNING_RATE,
    max_iter            = MAX_ITER,
    early_stopping      = True,
    validation_fraction = VAL_SPLIT,
    n_iter_no_change    = ES_PATIENCE,
    tol                 = 1e-5,
    random_state        = 42,
    verbose             = False,
)

arch_str = " -> ".join(str(x) for x in (input_dim,) + hidden_layers + (input_dim,))
n_params = sum([
    hidden_layers[0] * input_dim + hidden_layers[0],
    *[hidden_layers[i] * hidden_layers[i-1] + hidden_layers[i]
      for i in range(1, len(hidden_layers))],
    input_dim * hidden_layers[-1] + input_dim
])

mtb.add_message(
    f"Autoencoder architecture: {arch_str}\n"
    f"Approx parameters: {n_params:,}  |  "
    f"Bottleneck size: {bnk}  |  Activation: relu  |  Solver: adam"
)

# ---- 9. TRAIN ---------------------------------------------------------------
autoencoder.fit(X_tr, X_tr)   # target = input (reconstruction)

epochs_run  = autoencoder.n_iter_
final_loss  = autoencoder.loss_
loss_curve  = list(autoencoder.loss_curve_)
val_scores  = list(autoencoder.validation_scores_) if hasattr(autoencoder, "validation_scores_") else []

mtb.add_message(
    f"Training complete.  Iterations: {epochs_run}  |  "
    f"Final train loss: {final_loss:.6f}"
)

# ---- 10. SCORE ALL WINDOWS --------------------------------------------------
recon_all  = autoencoder.predict(all_windows)    # [n_windows, input_dim]
# Per-window MAE
window_mae = np.mean(np.abs(all_windows - recon_all), axis=1)

# Threshold from training windows
train_mae      = window_mae[train_mask]
threshold_val  = float(np.mean(train_mae) + THRESHOLD_K * np.std(train_mae))
window_flagged = (window_mae >= threshold_val).astype(int)
n_flagged_wins = int(window_flagged.sum())

# Map back to rows
row_flag_count = np.zeros(n_rows, dtype=int)
for w_idx, start in enumerate(window_starts):
    if window_flagged[w_idx]:
        row_flag_count[start:start + WINDOW_SIZE] += 1

row_anomaly    = (row_flag_count > 0).astype(int)
n_flagged_rows = int(row_anomaly.sum())

# Reshape reconstructions back to [n_windows, WINDOW_SIZE, n_features]
recon_3d     = recon_all.reshape(n_windows, WINDOW_SIZE, n_features)
windows_3d   = all_windows.reshape(n_windows, WINDOW_SIZE, n_features)

mtb.add_message(
    f"Threshold (mean + {THRESHOLD_K}sigma): {threshold_val:.6f}\n"
    f"Flagged windows : {n_flagged_wins} / {n_windows}\n"
    f"Flagged rows    : {n_flagged_rows} / {n_rows}  "
    f"({round(n_flagged_rows/n_rows*100, 2)} %)"
)

# =============================================================================
# HELPERS
# =============================================================================

def save_and_push(fig, fname):
    fp = os.path.join(plots_dir, fname)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    mtb.add_image(fp)
    plt.close(fig)

def bar_chart_height(n):
    return max(4, min(20, n * 0.45 + 1.5))

def smart_date_fmt(ax, ts):
    try:
        # Only apply date formatting if timestamps are real datetimes
        if isinstance(ts, pd.RangeIndex) or not hasattr(ts, "dtype"):
            return
        if not np.issubdtype(ts.dtype, np.datetime64):
            return
        span = (ts.max() - ts.min()).days
        fmt  = (mdates.DateFormatter("%m-%d %H:%M") if span <= 7
                else mdates.DateFormatter("%b %d") if span <= 90
                else mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_formatter(fmt)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    except Exception:
        pass

# =============================================================================
# OUTPUT SECTION
# =============================================================================

mtb.set_note(
    f"Autoencoder Anomaly Detection (scikit-learn)  |  "
    f"Rows: {n_rows}  |  Features: {n_features}  |  "
    f"Window: {WINDOW_SIZE}  |  Stride: {WINDOW_STRIDE}  |  "
    f"Threshold: {threshold_val:.4f}  |  "
    f"Flagged: {n_flagged_rows} rows ({round(n_flagged_rows/n_rows*100,1)} %)"
)

# ---- SECTION 1 — Config (text table) ----------------------------------------
mtb.add_table(
    columns=[
        ["Framework", "Model type",
         "Feature columns", "N features", "N rows",
         "Window size", "Window stride", "N windows total",
         "N training windows",
         "Architecture", "Approx parameters",
         "Activation", "Solver", "Initial LR",
         "Max iterations", "Early stopping patience",
         "Threshold rule", "Threshold K"],
        ["scikit-learn MLPRegressor (no PyTorch/TensorFlow)",
         "MLP Autoencoder — encoder/bottleneck/decoder",
         f"C{ts_idx+1} to C{max_feat_idx}",
         str(n_features), str(n_rows),
         str(WINDOW_SIZE), str(WINDOW_STRIDE), str(n_windows),
         str(int(train_mask.sum())),
         arch_str, f"{n_params:,}",
         "relu", "adam", str(LEARNING_RATE),
         str(MAX_ITER), str(ES_PATIENCE),
         "mean + K * std of train MAE", str(THRESHOLD_K)],
    ],
    headers=["Parameter", "Value"],
    title="Autoencoder Anomaly Detection — Configuration",
    footnote=(
        "Trained on normal windows only.  "
        "High reconstruction error on a window = pattern not seen in training = anomaly."
    )
)

# ---- SECTION 2 — Numeric results (separate table — no type mixing) ----------
mtb.add_table(
    columns=[
        ["Iterations run", "Final train loss (MSE)",
         "Train MAE mean", "Train MAE std",
         "Anomaly threshold",
         "Windows flagged", "Windows flagged (%)",
         "Rows flagged",    "Rows flagged (%)"],
        [float(epochs_run),
         round(float(final_loss), 6),
         round(float(np.mean(train_mae)), 6),
         round(float(np.std(train_mae)), 6),
         round(float(threshold_val), 6),
         float(n_flagged_wins),
         round(float(n_flagged_wins / n_windows * 100), 2),
         float(n_flagged_rows),
         round(float(n_flagged_rows / n_rows * 100), 2)],
    ],
    headers=["Metric", "Value"],
    title="Autoencoder — Detection Results",
    footnote=f"Threshold = mean + {THRESHOLD_K} * std of training window MAE."
)

# ---- PLOT 1 — Training Loss Curve -------------------------------------------
try:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(loss_curve, color="steelblue", linewidth=1.8, label="Train Loss (MSE)")
    if val_scores:
        ax2 = ax.twinx()
        ax2.plot(val_scores, color="darkorange", linewidth=1.8,
                 linestyle="--", label="Val R2")
        ax2.set_ylabel("Validation R2", color="darkorange")
        ax2.tick_params(axis="y", colors="darkorange")
        ax2.legend(loc="center right", fontsize=8)
    ax.axvline(epochs_run - 1, color="red", linewidth=1,
               linestyle=":", label=f"Stopped @ iter {epochs_run}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Train Loss (MSE)")
    ax.set_title(
        f"Autoencoder — Training Loss Curve\n"
        f"Iterations: {epochs_run}  |  Final loss: {final_loss:.5f}  |  "
        f"Framework: scikit-learn",
        fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_and_push(fig, "01_loss_curve.png")
except Exception as e:
    mtb.add_message(f"Loss curve skipped — {e}")

# ---- PLOT 2 — Reconstruction Error Distribution -----------------------------
try:
    fig, ax = plt.subplots(figsize=(10, 4))
    n_bins  = min(80, max(30, n_windows // 10))

    if has_labels and len(np.unique(window_labels[window_labels >= 0])) > 1:
        ax.hist(window_mae[window_labels == 0], bins=n_bins,
                alpha=0.65, color="steelblue", label="Normal windows",  density=True)
        ax.hist(window_mae[window_labels == 1], bins=n_bins,
                alpha=0.65, color="tomato",    label="Anomaly windows", density=True)
    else:
        ax.hist(window_mae, bins=n_bins, alpha=0.75,
                color="steelblue", label="All windows", density=True)

    ax.axvline(threshold_val, color="red", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold_val:.4f}")
    ax.set_xlabel("Window Reconstruction MAE")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Reconstruction Error Distribution\n"
        f"Threshold (mean+{THRESHOLD_K}sigma) = {threshold_val:.4f}  |  "
        f"{n_flagged_wins}/{n_windows} windows flagged",
        fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_and_push(fig, "02_error_distribution.png")
except Exception as e:
    mtb.add_message(f"Error distribution skipped — {e}")

# ---- PLOT 3 — Anomaly Timeline ----------------------------------------------
try:
    win_ts = [df["timestamp"].iloc[s] for s in window_starts]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(win_ts, window_mae, color="steelblue", linewidth=0.8,
            alpha=0.7, label="Window MAE")
    ax.axhline(threshold_val, color="red", linewidth=1.5, linestyle="--",
               label=f"Threshold = {threshold_val:.4f}")
    ax.fill_between(win_ts, window_mae, threshold_val,
                    where=np.array(window_mae) >= threshold_val,
                    color="tomato", alpha=0.4, label="Anomalous region")
    ax.set_ylabel("Reconstruction MAE")
    ax.set_title("Autoencoder — Anomaly Timeline", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.fill_between(win_ts, window_flagged, step="pre",
                     color="tomato", alpha=0.7, label="Flagged")
    ax2.set_ylabel("Flag")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Anomaly"])
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=8)
    try:
        smart_date_fmt(ax2, df["timestamp"])
    except Exception:
        pass

    fig.tight_layout()
    save_and_push(fig, "03_anomaly_timeline.png")
except Exception as e:
    mtb.add_message(f"Timeline skipped — {e}")

# ---- PLOT 4 — First Feature with Anomaly Overlay ----------------------------
try:
    feat_show = feature_names[0]
    fig, ax   = plt.subplots(figsize=(14, 4))
    ax.plot(df["timestamp"], df[feat_show],
            color="steelblue", linewidth=0.7, label=feat_show)

    in_anom, start_anom = False, None
    for i, flag in enumerate(row_anomaly):
        ts_i = df["timestamp"].iloc[i]
        if flag and not in_anom:
            start_anom, in_anom = ts_i, True
        elif not flag and in_anom:
            ax.axvspan(start_anom, ts_i, color="tomato", alpha=0.3)
            in_anom = False
    if in_anom:
        ax.axvspan(start_anom, df["timestamp"].iloc[-1], color="tomato", alpha=0.3)

    ax.set_ylabel(feat_show)
    ax.set_title(
        f"Feature '{feat_show}'  |  Red shading = anomalous periods\n"
        f"({n_flagged_rows} / {n_rows} rows flagged  "
        f"= {round(n_flagged_rows/n_rows*100, 1)} %)",
        fontweight="bold"
    )
    try:
        smart_date_fmt(ax, df["timestamp"])
    except Exception:
        pass
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_and_push(fig, "04_feature_anomaly_overlay.png")
except Exception as e:
    mtb.add_message(f"Feature overlay skipped — {e}")

# ---- PLOT 5 — All Features Small Multiples ----------------------------------
try:
    n_feat_show = min(n_features, 6)
    fig, axes   = plt.subplots(n_feat_show, 1,
                                figsize=(14, 2.5 * n_feat_show), sharex=True)
    if n_feat_show == 1:
        axes = [axes]

    for ax, feat in zip(axes, feature_names[:n_feat_show]):
        ax.plot(df["timestamp"], df[feat], color="steelblue",
                linewidth=0.6, alpha=0.85)
        ax.fill_between(df["timestamp"],
                        df[feat].min(), df[feat].max(),
                        where=row_anomaly.astype(bool),
                        color="tomato", alpha=0.25)
        ax.set_ylabel(feat, fontsize=8)
        ax.grid(True, alpha=0.15)

    try:
        smart_date_fmt(axes[-1], df["timestamp"])
    except Exception:
        pass

    fig.suptitle(
        f"All Features — Anomalous Periods Highlighted (red)\n"
        f"Top {n_feat_show} of {n_features} features shown",
        fontweight="bold"
    )
    fig.tight_layout()
    save_and_push(fig, "05_all_features_overview.png")
except Exception as e:
    mtb.add_message(f"All-features overview skipped — {e}")

# ---- PLOT 6 — Sample Window Reconstructions ---------------------------------
try:
    normal_wins = np.where(window_flagged == 0)[0]
    anom_wins   = np.where(window_flagged == 1)[0]
    n_n         = min(MAX_RECON_PLOTS // 2, len(normal_wins))
    n_a         = min(MAX_RECON_PLOTS // 2, len(anom_wins))
    show_ids    = list(normal_wins[:n_n]) + list(anom_wins[:n_a])

    if show_ids:
        n_cols     = 2
        n_rows_fig = int(math.ceil(len(show_ids) / n_cols))
        fig, axes  = plt.subplots(n_rows_fig, n_cols,
                                   figsize=(12, 3.5 * n_rows_fig), squeeze=False)
        feat_idx   = 0
        feat_lbl   = feature_names[feat_idx]

        for k, w_idx in enumerate(show_ids):
            ax    = axes[k // n_cols][k % n_cols]
            inp   = windows_3d[w_idx, :, feat_idx]
            rec   = recon_3d[w_idx, :, feat_idx]
            err   = window_mae[w_idx]
            flag  = window_flagged[w_idx]
            color = "tomato" if flag else "steelblue"
            lbl   = "ANOMALY" if flag else "Normal"

            ax.plot(inp, color="black", linewidth=1.5, label="Input")
            ax.plot(rec, color=color,   linewidth=1.5,
                    linestyle="--", label="Reconstruction")
            ax.fill_between(range(WINDOW_SIZE), inp, rec,
                            alpha=0.2, color=color)
            ax.set_title(
                f"Window #{w_idx}  [{lbl}]  MAE={err:.4f}",
                fontsize=9, fontweight="bold",
                color="darkred" if flag else "navy"
            )
            ax.set_xlabel("Time step within window")
            ax.set_ylabel(feat_lbl)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

        for k in range(len(show_ids), n_rows_fig * n_cols):
            axes[k // n_cols][k % n_cols].set_visible(False)

        fig.suptitle(
            f"Sample Window Reconstructions — Feature: '{feat_lbl}'\n"
            f"Shaded area = reconstruction error  |  Window = {WINDOW_SIZE} rows",
            fontweight="bold"
        )
        fig.tight_layout()
        save_and_push(fig, "06_sample_reconstructions.png")
except Exception as e:
    mtb.add_message(f"Sample reconstructions skipped — {e}")

# ---- PLOT 7 — Feature Error Heatmap -----------------------------------------
try:
    anom_win_idx = np.where(window_flagged == 1)[0]
    if len(anom_win_idx) > 0:
        n_show    = min(40, len(anom_win_idx))
        show_idx  = anom_win_idx[:n_show]
        # Per-feature MAE: mean over timesteps within each window
        feat_errs = np.mean(
            np.abs(windows_3d[show_idx] - recon_3d[show_idx]), axis=1
        )   # shape: (n_show, n_features)

        fig, ax = plt.subplots(
            figsize=(max(8, n_features * 0.7), max(4, n_show * 0.3 + 2))
        )
        im = ax.imshow(feat_errs, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f"Win #{i}" for i in show_idx], fontsize=7)
        ax.set_title(
            f"Feature-Level Reconstruction Error — Flagged Windows\n"
            f"({n_show} of {len(anom_win_idx)} shown)  |  "
            f"Brighter = higher error in that feature",
            fontweight="bold"
        )
        plt.colorbar(im, ax=ax, label="MAE per feature")
        fig.tight_layout()
        save_and_push(fig, "07_feature_error_heatmap.png")
except Exception as e:
    mtb.add_message(f"Feature heatmap skipped — {e}")

# ---- PLOTS 8 & 9 — ROC + PR (supervised only) -------------------------------
if has_labels and len(np.unique(window_labels[window_labels >= 0])) > 1:
    try:
        valid  = window_labels >= 0
        y_true = window_labels[valid]
        scores = window_mae[valid]
        y_pred = window_flagged[valid]

        roc_auc = roc_auc_score(y_true, scores)
        pr_auc  = average_precision_score(y_true, scores)

        fpr, tpr, _ = roc_curve(y_true, scores)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="steelblue", linewidth=2,
                label=f"ROC AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        save_and_push(fig, "08_roc_curve.png")

        prec, rec, _ = precision_recall_curve(y_true, scores)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, color="darkorange", linewidth=2,
                label=f"PR AUC = {pr_auc:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve", fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        save_and_push(fig, "09_pr_curve.png")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        prec_v = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_v  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_v   = (2 * prec_v * rec_v / (prec_v + rec_v)
                  if (prec_v + rec_v) > 0 else 0)

        mtb.add_table(
            columns=[
                ["ROC-AUC", "PR-AUC",
                 "True Positives", "True Negatives",
                 "False Positives", "False Negatives",
                 "Precision", "Recall", "F1 Score"],
                [round(roc_auc, 4), round(pr_auc, 4),
                 float(tp), float(tn), float(fp), float(fn),
                 round(prec_v, 4), round(rec_v, 4), round(f1_v, 4)],
            ],
            headers=["Metric", "Value"],
            title="Supervised Evaluation Metrics (Window-Level)",
            footnote=(
                f"Threshold = {threshold_val:.6f}  |  "
                "A window is anomalous if ANY row in it has label=1."
            )
        )
    except Exception as e:
        mtb.add_message(f"ROC/PR metrics skipped — {e}")

gc.collect()

# =============================================================================
# SECTION 3 — ENHANCED ALARM LOG
# =============================================================================
# Builds a rich per-period alarm table including:
#   • Severity tier  (Advisory / Warning / Critical)
#   • Top contributing sensor + its % share of total window MAE
#   • Sensor direction at alarm onset  (Rising / Falling / Flat)
#   • Suppression flag (whether harvester or other known mode was active)
#   • Consecutive-window filter applied (noise windows removed)
# =============================================================================

try:
    # ------------------------------------------------------------------
    # STEP A — Per-sensor MAE contribution for every flagged window
    #          feat_errs shape: (n_flagged_windows, n_features)
    # ------------------------------------------------------------------
    anom_win_idx = np.where(window_flagged == 1)[0]

    # Sensor contribution % for each flagged window
    if len(anom_win_idx) > 0:
        feat_errs_all = np.mean(
            np.abs(windows_3d[anom_win_idx] - recon_3d[anom_win_idx]), axis=1
        )   # (n_flagged, n_features)
        feat_errs_pct = feat_errs_all / (feat_errs_all.sum(axis=1, keepdims=True) + 1e-9)
    else:
        feat_errs_all = np.zeros((0, n_features))
        feat_errs_pct = np.zeros((0, n_features))

    # ------------------------------------------------------------------
    # STEP B — Suppression mask per window
    #          If SUPPRESSION_FEATURE exists in the data and equals 1
    #          for >50% of the window rows, mark as suppressed.
    # ------------------------------------------------------------------
    supp_col_exists = (
        SUPPRESSION_FEATURE is not None
        and SUPPRESSION_FEATURE in feature_names
    )
    supp_feat_idx = feature_names.index(SUPPRESSION_FEATURE) if supp_col_exists else None

    def is_suppressed(w_idx):
        if not supp_col_exists:
            return False
        start = window_starts[w_idx]
        window_rows = df[SUPPRESSION_FEATURE].iloc[start:start + WINDOW_SIZE].values
        return float(np.mean(window_rows > 0.5)) >= 0.5

    # ------------------------------------------------------------------
    # STEP C — Severity tier per flagged window
    # ------------------------------------------------------------------
    def severity_tier(mae):
        if mae >= threshold_val * SEVERITY_CRITICAL_MULT:
            return "Critical"
        elif mae >= threshold_val * SEVERITY_WARNING_MULT:
            return "Warning"
        else:
            return "Advisory"

    # ------------------------------------------------------------------
    # STEP D — Sensor direction at the START of each flagged window
    #          Compare first-quarter mean vs last-quarter mean of window.
    # ------------------------------------------------------------------
    def sensor_direction(w_idx, feat_idx):
        start  = window_starts[w_idx]
        q      = max(1, WINDOW_SIZE // 4)
        early  = scaled[start:start + q, feat_idx].mean()
        late   = scaled[start + WINDOW_SIZE - q:start + WINDOW_SIZE, feat_idx].mean()
        diff   = late - early
        if diff > 0.1:
            return "Rising"
        elif diff < -0.1:
            return "Falling"
        return "Flat"

    # ------------------------------------------------------------------
    # STEP E — Consecutive window filter
    #          Group consecutive flagged window indices; keep only groups
    #          of size >= MIN_CONSECUTIVE_WINDOWS.
    # ------------------------------------------------------------------
    def get_consecutive_groups(flagged_idx_arr, min_run):
        """Return list of lists, each a run of consecutive flagged window indices."""
        if len(flagged_idx_arr) == 0:
            return []
        groups, current = [], [flagged_idx_arr[0]]
        for i in flagged_idx_arr[1:]:
            if i == current[-1] + 1:
                current.append(i)
            else:
                groups.append(current)
                current = [i]
        groups.append(current)
        return [g for g in groups if len(g) >= min_run]

    consec_groups = get_consecutive_groups(
        list(anom_win_idx), MIN_CONSECUTIVE_WINDOWS
    )
    n_noise_wins  = len(anom_win_idx) - sum(len(g) for g in consec_groups)

    if n_noise_wins > 0:
        mtb.add_message(
            f"Consecutive-window filter (MIN={MIN_CONSECUTIVE_WINDOWS}): "
            f"{n_noise_wins} isolated window(s) removed as likely noise.  "
            f"{len(consec_groups)} alarm event(s) remain."
        )

    # ------------------------------------------------------------------
    # STEP F — Map grouped windows back to row-level alarm periods
    # ------------------------------------------------------------------
    alarm_periods = []
    for group in consec_groups:
        first_w   = group[0]
        last_w    = group[-1]
        row_start = window_starts[first_w]
        row_end   = window_starts[last_w] + WINDOW_SIZE - 1
        row_end   = min(row_end, n_rows - 1)

        # Max MAE in the group → severity is determined by worst window
        group_maes    = window_mae[group]
        peak_mae      = float(group_maes.max())
        peak_win_pos  = int(np.argmax(group_maes))
        peak_win_idx  = group[peak_win_pos]     # absolute window index

        # Map peak_win_idx to position within anom_win_idx for feat_errs lookup
        anom_pos = np.searchsorted(anom_win_idx, peak_win_idx)
        if anom_pos < len(anom_win_idx) and anom_win_idx[anom_pos] == peak_win_idx:
            top_feat_idx  = int(np.argmax(feat_errs_pct[anom_pos]))
            top_feat_pct  = round(float(feat_errs_pct[anom_pos, top_feat_idx]) * 100, 1)
            top_feat_name = feature_names[top_feat_idx]
            direction     = sensor_direction(peak_win_idx, top_feat_idx)
        else:
            top_feat_name = "N/A"
            top_feat_pct  = 0.0
            direction     = "N/A"

        suppressed = any(is_suppressed(w) for w in group)

        alarm_periods.append({
            "row_start"  : row_start,
            "row_end"    : row_end,
            "duration"   : row_end - row_start + 1,
            "n_windows"  : len(group),
            "peak_mae"   : round(peak_mae, 4),
            "severity"   : severity_tier(peak_mae),
            "top_sensor" : top_feat_name,
            "sensor_pct" : top_feat_pct,
            "direction"  : direction,
            "suppressed" : "Yes" if suppressed else "No",
        })

    # ------------------------------------------------------------------
    # STEP G — Enhanced alarm log table (all columns text-safe)
    # ------------------------------------------------------------------
    if alarm_periods:
        mtb.add_table(
            columns=[
                list(range(1, len(alarm_periods) + 1)),
                [str(df["timestamp"].iloc[p["row_start"]]) for p in alarm_periods],
                [str(df["timestamp"].iloc[p["row_end"]])   for p in alarm_periods],
                [p["duration"]   for p in alarm_periods],
                [p["n_windows"]  for p in alarm_periods],
                [p["peak_mae"]   for p in alarm_periods],
            ],
            headers=["#", "Start", "End", "Duration (rows)",
                     "Windows", "Peak MAE"],
            title=f"Enhanced Alarm Log — {len(alarm_periods)} alarm event(s) after noise filter",
            footnote=(
                f"Consecutive filter: min {MIN_CONSECUTIVE_WINDOWS} windows.  "
                f"Noise windows removed: {n_noise_wins}.  "
                f"Threshold = {threshold_val:.4f}"
            )
        )

        # Severity + sensor columns — text-only table (avoids type mixing)
        mtb.add_table(
            columns=[
                list(range(1, len(alarm_periods) + 1)),
                [p["severity"]   for p in alarm_periods],
                [p["top_sensor"] for p in alarm_periods],
                [PROCESS_SECTION_MAP.get(p["top_sensor"], "Unknown")
                 for p in alarm_periods],
                [p["direction"]  for p in alarm_periods],
                [p["suppressed"] for p in alarm_periods],
            ],
            headers=["#", "Severity", "Top sensor", "Process section",
                     "Direction", "Suppressed"],
            title="Alarm Detail — Severity, Process Section and Root Cause Sensor",
            footnote=(
                f"Severity: Advisory = MAE < {SEVERITY_WARNING_MULT}x threshold  |  "
                f"Warning = {SEVERITY_WARNING_MULT}x-{SEVERITY_CRITICAL_MULT}x  |  "
                f"Critical = >{SEVERITY_CRITICAL_MULT}x.  "
                f"Process sections: Forming / Dryer / Blending / Press / Mechanical."
            )
        )

        # Sensor % contribution table — numeric
        mtb.add_table(
            columns=[
                list(range(1, len(alarm_periods) + 1)),
                [p["top_sensor"] for p in alarm_periods],
                [p["sensor_pct"] for p in alarm_periods],
            ],
            headers=["#", "Top sensor", "% of alarm MAE"],
            title="Sensor Contribution — Primary Driver per Alarm Event",
            footnote=(
                "% of total reconstruction error attributable to the leading sensor.  "
                "Use this to prioritise which tag to investigate first."
            )
        )
    else:
        mtb.add_message(
            "No alarm events survived the consecutive-window filter.  "
            f"Try lowering MIN_CONSECUTIVE_WINDOWS (currently {MIN_CONSECUTIVE_WINDOWS}) "
            f"or THRESHOLD_K (currently {THRESHOLD_K})."
        )

except Exception as e:
    mtb.add_message(f"Enhanced alarm log skipped — {e}")

# =============================================================================
# SECTION 4 — ENHANCED FEATURE HEATMAP WITH % CONTRIBUTION ANNOTATION
# =============================================================================
# Upgrades Plot 07: adds numeric % labels inside each heatmap cell so the
# engineer can read exact contribution without interpreting colour alone.
# =============================================================================

try:
    anom_win_idx = np.where(window_flagged == 1)[0]
    if len(anom_win_idx) > 0 and len(feat_errs_pct) > 0:
        n_show   = min(40, len(anom_win_idx))
        show_idx = anom_win_idx[:n_show]
        pct_show = feat_errs_pct[:n_show]          # (n_show, n_features)
        err_show = feat_errs_all[:n_show]

        fig, ax = plt.subplots(
            figsize=(max(10, n_features * 0.85), max(5, n_show * 0.35 + 2))
        )
        im = ax.imshow(err_show, aspect="auto", cmap="YlOrRd")

        # Annotate each cell with its % contribution
        for row_i in range(n_show):
            for col_j in range(n_features):
                pct_val  = pct_show[row_i, col_j] * 100
                cell_val = err_show[row_i, col_j]
                # Use dark text on light cells, white text on dark cells
                cell_norm = cell_val / (err_show.max() + 1e-9)
                txt_color = "white" if cell_norm > 0.55 else "black"
                if pct_val >= 5.0:   # only annotate if contribution is meaningful
                    ax.text(col_j, row_i, f"{pct_val:.0f}%",
                            ha="center", va="center", fontsize=7,
                            color=txt_color, fontweight="bold")

        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_show))

        # Y-axis labels: include severity tier
        y_labels = []
        for abs_w in show_idx:
            mae_w  = window_mae[abs_w]
            sev    = severity_tier(mae_w)
            sev_ch = {"Critical": "C", "Warning": "W", "Advisory": "A"}[sev]
            y_labels.append(f"Win#{abs_w} [{sev_ch}] {mae_w:.3f}")

        ax.set_yticklabels(y_labels, fontsize=7)
        plt.colorbar(im, ax=ax, label="MAE per feature")
        ax.set_title(
            f"Feature Contribution Heatmap — Flagged Windows\n"
            f"Numbers = % of alarm MAE  |  C=Critical  W=Warning  A=Advisory  |  "
            f"{n_show} of {len(anom_win_idx)} shown",
            fontweight="bold"
        )
        fig.tight_layout()
        save_and_push(fig, "07b_feature_heatmap_annotated.png")
except Exception as e:
    mtb.add_message(f"Annotated heatmap skipped — {e}")

# =============================================================================
# SECTION 4b — PROCESS SECTION CONTRIBUTION BAR CHART
# =============================================================================
# Groups per-sensor MAE into the 5 process sections (Forming, Dryer,
# Blending, Press, Mechanical) for each flagged window.
# Instantly shows which section of the OSB line is causing the alarm.
# =============================================================================

try:
    if len(anom_win_idx) > 0 and len(feat_errs_pct) > 0:
        sections      = ["Forming", "Dryer", "Blending", "Press", "Mechanical"]
        section_colors = {
            "Forming"   : "steelblue",
            "Dryer"     : "darkorange",
            "Blending"  : "seagreen",
            "Press"     : "tomato",
            "Mechanical": "mediumpurple",
        }

        # Build section-level contribution matrix
        n_show_sec   = min(40, len(anom_win_idx))
        show_idx_sec = anom_win_idx[:n_show_sec]
        pct_sec      = feat_errs_pct[:n_show_sec]   # (n_show, n_features)

        # Map each feature to its section index
        section_pct = np.zeros((n_show_sec, len(sections)))
        for fi, feat in enumerate(feature_names):
            sec = PROCESS_SECTION_MAP.get(feat, "Mechanical")
            si  = sections.index(sec) if sec in sections else 4
            section_pct[:, si] += pct_sec[:, fi]

        # Stacked bar chart — each bar = one flagged window
        fig, ax = plt.subplots(figsize=(14, 5))
        bottoms = np.zeros(n_show_sec)
        for si, sec in enumerate(sections):
            ax.bar(range(n_show_sec), section_pct[:, si],
                   bottom=bottoms, label=sec,
                   color=section_colors[sec], alpha=0.85, width=0.75)
            bottoms += section_pct[:, si]

        # Severity tier markers above each bar
        for bar_i, abs_w in enumerate(show_idx_sec):
            mae_w = window_mae[abs_w]
            sev   = severity_tier(mae_w)
            mk    = {"Critical": "*", "Warning": "^", "Advisory": "."}[sev]
            col   = {"Critical": "red", "Warning": "darkorange",
                     "Advisory": "gray"}[sev]
            ax.plot(bar_i, 1.03, mk, color=col, markersize=8,
                    transform=ax.get_xaxis_transform())

        ax.set_xticks(range(n_show_sec))
        ax.set_xticklabels(
            [f"#{anom_win_idx[i]}" for i in range(n_show_sec)],
            rotation=45, ha="right", fontsize=7
        )
        ax.set_ylabel("Fraction of alarm MAE")
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
        )
        ax.legend(loc="upper right", fontsize=9, title="Process section")
        ax.set_title(
            "Process Section Contribution — Flagged Windows\n"
            "Markers above bars: * Critical   ^ Warning   . Advisory",
            fontweight="bold"
        )
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        save_and_push(fig, "07c_section_contribution.png")

        # Section summary table — which section caused most alarms
        section_alarm_count = {s: 0 for s in sections}
        section_avg_pct     = {s: [] for s in sections}
        for bar_i in range(n_show_sec):
            dominant_sec = sections[int(np.argmax(section_pct[bar_i]))]
            section_alarm_count[dominant_sec] += 1
            for si, sec in enumerate(sections):
                section_avg_pct[sec].append(section_pct[bar_i, si])

        mtb.add_table(
            columns=[
                sections,
                [section_alarm_count[s] for s in sections],
                [round(float(np.mean(section_avg_pct[s])) * 100, 1)
                 for s in sections],
            ],
            headers=["Process section", "Alarms led", "Avg MAE share (%)"],
            title="Process Section Alarm Summary",
            footnote=(
                "Alarms led = number of flagged windows where this section "
                "contributed the largest share of reconstruction error.  "
                "Avg MAE share = average % contribution across all flagged windows."
            )
        )

except Exception as e:
    mtb.add_message(f"Process section chart skipped — {e}")

# =============================================================================
# SECTION 5 — ROLLING MAE EQUIPMENT HEALTH TREND
# =============================================================================
# Computes a rolling average of per-sensor reconstruction error over
# ROLLING_HEALTH_WINDOWS windows. A rising trend on any sensor over weeks
# signals developing degradation — even before an alarm threshold is crossed.
# =============================================================================

try:
    # Per-sensor MAE for EVERY window (not just flagged ones)
    # windows_3d shape: (n_windows, WINDOW_SIZE, n_features)
    # recon_3d   shape: (n_windows, WINDOW_SIZE, n_features)
    per_sensor_mae_all = np.mean(
        np.abs(windows_3d - recon_3d), axis=1
    )   # (n_windows, n_features)

    win_ts_arr = np.array([df["timestamp"].iloc[s] for s in window_starts])

    n_roll = min(ROLLING_HEALTH_WINDOWS, n_windows // 2)
    if n_roll >= 2:
        # Rolling mean per sensor using pandas for convenience
        sensor_mae_df = pd.DataFrame(per_sensor_mae_all, columns=feature_names)
        rolling_mae   = sensor_mae_df.rolling(window=n_roll, min_periods=1).mean()

        n_feat_health = min(n_features, 6)   # show top 6 sensors max
        fig, axes = plt.subplots(
            n_feat_health, 1,
            figsize=(14, 2.5 * n_feat_health),
            sharex=True
        )
        if n_feat_health == 1:
            axes = [axes]

        for ax_i, feat in enumerate(feature_names[:n_feat_health]):
            raw_vals  = per_sensor_mae_all[:, ax_i]
            roll_vals = rolling_mae[feat].values

            ax = axes[ax_i]
            ax.plot(win_ts_arr, raw_vals,  color="lightsteelblue",
                    linewidth=0.5, alpha=0.6, label="Window MAE")
            ax.plot(win_ts_arr, roll_vals, color="steelblue",
                    linewidth=1.8, label=f"{n_roll}-window rolling avg")
            ax.axhline(threshold_val, color="red", linewidth=0.8,
                       linestyle="--", alpha=0.6, label="Alarm threshold")

            # Shade alarm windows
            for w_i in anom_win_idx:
                ax.axvspan(win_ts_arr[w_i],
                           win_ts_arr[min(w_i + 1, n_windows - 1)],
                           color="tomato", alpha=0.2)

            ax.set_ylabel(feat, fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.15)

        try:
            smart_date_fmt(axes[-1], df["timestamp"])
        except Exception:
            pass

        fig.suptitle(
            f"Equipment Health Trend — Rolling MAE per Sensor\n"
            f"Rising trend = developing degradation even before alarm fires  |  "
            f"Red shading = flagged windows  |  Rolling window = {n_roll}",
            fontweight="bold"
        )
        fig.tight_layout()
        save_and_push(fig, "10_rolling_health_trend.png")

        # Health trend summary table — compare first-half vs second-half rolling MAE
        half = n_windows // 2
        trend_labels, trend_first, trend_second, trend_direction = [], [], [], []
        for feat in feature_names:
            first_h  = round(float(rolling_mae[feat].iloc[:half].mean()), 5)
            second_h = round(float(rolling_mae[feat].iloc[half:].mean()), 5)
            pct_chg  = ((second_h - first_h) / (first_h + 1e-9)) * 100
            if pct_chg > 5:
                trend_dir = "Degrading"
            elif pct_chg < -5:
                trend_dir = "Improving"
            else:
                trend_dir = "Stable"
            trend_labels.append(feat)
            trend_first.append(first_h)
            trend_second.append(second_h)
            trend_direction.append(trend_dir)

        mtb.add_table(
            columns=[trend_labels, trend_first, trend_second, trend_direction],
            headers=["Sensor", "First-half avg MAE", "Second-half avg MAE", "Trend"],
            title="Sensor Health Trend — First Half vs Second Half of Dataset",
            footnote=(
                "Degrading = second-half rolling MAE is >5% higher than first-half.  "
                "A degrading sensor may need calibration or maintenance even if no alarm was raised."
            )
        )

except Exception as e:
    mtb.add_message(f"Rolling health trend skipped — {e}")

# =============================================================================
# SECTION 6 — LEAD SENSOR ANALYSIS
# =============================================================================
# For each alarm event, looks back N windows BEFORE the threshold was crossed
# and identifies which sensor's MAE was already rising earliest.
# That sensor is your early warning indicator for that fault type.
# =============================================================================

try:
    LOOKBACK_WINDOWS = 4   # how many windows before alarm onset to examine

    lead_events, lead_sensors, lead_advances = [], [], []

    for grp_idx, group in enumerate(consec_groups):
        first_flagged_w = group[0]
        if first_flagged_w < LOOKBACK_WINDOWS:
            continue   # not enough history before this event

        # Per-sensor MAE in the LOOKBACK window just before alarm
        lookback_range = range(
            first_flagged_w - LOOKBACK_WINDOWS, first_flagged_w
        )
        pre_alarm_mae = per_sensor_mae_all[list(lookback_range), :]
        # Slope of each sensor's MAE in the lookback period
        # (positive slope = rising toward alarm)
        slopes = np.polyfit(
            np.arange(LOOKBACK_WINDOWS),
            pre_alarm_mae, deg=1
        )[0]   # shape: (n_features,)

        lead_feat_idx  = int(np.argmax(slopes))
        lead_feat_name = feature_names[lead_feat_idx]
        lead_slope     = round(float(slopes[lead_feat_idx]), 5)

        lead_events.append(grp_idx + 1)
        lead_sensors.append(lead_feat_name)
        lead_advances.append(lead_slope)

    if lead_events:
        mtb.add_table(
            columns=[lead_events, lead_sensors, lead_advances],
            headers=["Alarm #", "Lead sensor", "MAE slope before alarm"],
            title=f"Lead Sensor Analysis — Early Warning Indicators ({LOOKBACK_WINDOWS}-window lookback)",
            footnote=(
                "Lead sensor = the tag whose reconstruction error was rising fastest "
                f"in the {LOOKBACK_WINDOWS} windows before the alarm threshold was crossed.  "
                "Monitor this sensor first for early fault detection."
            )
        )

        # Count which sensor leads most often
        from collections import Counter
        lead_counts = Counter(lead_sensors)
        most_common_lead = lead_counts.most_common(1)[0]
        mtb.add_message(
            f"Most frequent lead sensor across all alarm events: "
            f"'{most_common_lead[0]}' — led {most_common_lead[1]} of "
            f"{len(consec_groups)} alarm(s).  "
            f"Consider adding a dedicated early-warning monitor on this tag."
        )

except Exception as e:
    mtb.add_message(f"Lead sensor analysis skipped — {e}")

gc.collect()