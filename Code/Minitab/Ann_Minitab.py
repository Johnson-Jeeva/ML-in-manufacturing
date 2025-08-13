# ann_minitab.py
from mtbpy import mtbpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Connect to Minitab
# ---------------------------
mtb = mtbpy.mtb_instance()

# ---------------------------
# Parse arguments
#   Required:
#     sys.argv[1] -> target_col (e.g., "C5")
#     sys.argv[2] -> max_col    (e.g., "C20")
#   Optional flags:
#     --epochs N
#     --batchsize N
#     --patience N
#     --testsize FLOAT (e.g., 0.3)
#     --seed N
# ---------------------------
if len(sys.argv) < 3:
    mtb.add_message("Usage: python ann_minitab.py <target_col> <max_col> [--epochs N] [--batchsize N] [--patience N] [--testsize F] [--seed N]")
    sys.exit(1)

target_col = sys.argv[1]  # e.g., "C5"
max_col    = sys.argv[2]  # e.g., "C20"

# Defaults to mimic your code
epochs     = 300
batch_size = 64
patience   = 15
test_size  = 0.30
seed       = 42

# Optional overrides
for i, arg in enumerate(sys.argv):
    if arg == "--epochs" and i+1 < len(sys.argv):
        epochs = int(sys.argv[i+1])
    elif arg == "--batchsize" and i+1 < len(sys.argv):
        batch_size = int(sys.argv[i+1])
    elif arg == "--patience" and i+1 < len(sys.argv):
        patience = int(sys.argv[i+1])
    elif arg == "--testsize" and i+1 < len(sys.argv):
        test_size = float(sys.argv[i+1])
    elif arg == "--seed" and i+1 < len(sys.argv):
        seed = int(sys.argv[i+1])

# ---------------------------
# Build list of columns C1..Cmax (predictors exclude target)
# ---------------------------
max_idx      = int(max_col[1:])  # "C20" -> 20
all_columns  = [f"C{i}" for i in range(1, max_idx + 1)]
predict_cols = [c for c in all_columns if c != target_col]

# ---------------------------
# Pull data from Minitab
# ---------------------------
# Target
try:
    y = np.array(mtb.get_column(target_col), dtype=float)
except Exception as e:
    mtb.add_message(f"Error: could not read target column {target_col}: {e}")
    sys.exit(1)

# Predictors
X_data = []
valid_cols = []
for col in predict_cols:
    try:
        values = np.array(mtb.get_column(col), dtype=float)
        X_data.append(values)
        valid_cols.append(col)
    except Exception:
        mtb.add_message(f"Warning: Column {col} not found or non-numeric. Skipping.")

if len(valid_cols) == 0:
    mtb.add_message("Error: No valid predictor columns found.")
    sys.exit(1)

# Align lengths and make DataFrame
min_len = min(len(y), *(len(v) for v in X_data))
X_np = np.vstack([v[:min_len] for v in X_data]).T  # shape (n, p)
y   = y[:min_len]
X_df = pd.DataFrame(X_np, columns=valid_cols)

# Drop rows with NaNs if any
mask = np.isfinite(X_df).all(axis=1) & np.isfinite(y)
X_df = X_df.loc[mask].reset_index(drop=True)
y    = y[mask]

# ---------------------------
# Split data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=test_size, random_state=seed
)

# ---------------------------
# Scale features and target
# ---------------------------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))  # (n_train,1)
y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1))       # (n_test,1)

# ---------------------------
# Build Advanced ANN (exact architecture)
# 512 -> 256 -> 128 -> 64 -> 1
# ReLU for hidden, Linear for output
# ---------------------------
model_ann = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64,  activation='relu'),
    Dense(1,   activation='linear')
])

model_ann.compile(optimizer='adam', loss='mse')

# Early stopping (monitor training loss to match your prior script)
early_stop = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)

# ---------------------------
# Train model
# ---------------------------
history = model_ann.fit(
    X_train_scaled, y_train_scaled,
    epochs=epochs,
    batch_size=batch_size,
    verbose=0,
    callbacks=[early_stop]
)

# ---------------------------
# Predictions (inverse-transform back to original scale)
# ---------------------------
y_train_pred_scaled = model_ann.predict(X_train_scaled)          # (n_train,1)
y_test_pred_scaled  = model_ann.predict(X_test_scaled)           # (n_test,1)

y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).ravel()
y_test_pred  = scaler_y.inverse_transform(y_test_pred_scaled).ravel()

# ---------------------------
# Metrics
# ---------------------------
train_r2   = r2_score(y_train, y_train_pred)
test_r2    = r2_score(y_test,  y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test,  y_test_pred))

# ---------------------------
# Write summary to Minitab Output pane
# ---------------------------
mtb.add_message("Model: Keras ANN (Regression)")
mtb.add_message(f"Target column: {target_col}")
mtb.add_message(f"Predictors: {len(valid_cols)} columns from C1..{max_col} (excluding target)")
mtb.add_message(f"Architecture: [Dense(512, relu) -> Dense(256, relu) -> Dense(128, relu) -> Dense(64, relu) -> Dense(1, linear)]")
mtb.add_message(f"epochs={epochs}, batch_size={batch_size}, patience={patience}, test_size={test_size}, seed={seed}")
mtb.add_message(f"Train R^2: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
mtb.add_message(f"Test  R^2: {test_r2:.4f}  | RMSE: {test_rmse:.4f}")

# ---------------------------
# Plot 1: Training Loss curve
# ---------------------------
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.title("ANN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
loss_path = "ANN_training_loss.png"
plt.savefig(loss_path, dpi=300, bbox_inches="tight")
mtb.add_image(loss_path)

# ---------------------------
# Plot 2: Actual vs Predicted (Training)
# ---------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_train_pred, y_train, s=10)
lo = min(np.min(y_train), np.min(y_train_pred))
hi = max(np.max(y_train), np.max(y_train_pred))
plt.plot([lo, hi], [lo, hi])
plt.title("Training: Actual vs Predicted (ANN)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
train_ap_path = "ANN_train_actual_vs_pred.png"
plt.savefig(train_ap_path, dpi=300, bbox_inches="tight")
mtb.add_image(train_ap_path)

# ---------------------------
# Plot 3: Actual vs Predicted (Test)
# ---------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test_pred, y_test, s=10)
lo = min(np.min(y_test), np.min(y_test_pred))
hi = max(np.max(y_test), np.max(y_test_pred))
plt.plot([lo, hi], [lo, hi])
plt.title("Test: Actual vs Predicted (ANN)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
test_ap_path = "ANN_test_actual_vs_pred.png"
plt.savefig(test_ap_path, dpi=300, bbox_inches="tight")
mtb.add_image(test_ap_path)

# ---------------------------
# Plot 4: Residuals vs Predicted (Test)
# ---------------------------
residuals = y_test - y_test_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_test_pred, residuals, s=10)
plt.axhline(0)
plt.title("Residuals vs Predicted (Test)")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.tight_layout()
resid_path = "ANN_test_residuals.png"
plt.savefig(resid_path, dpi=300, bbox_inches="tight")
mtb.add_image(resid_path)