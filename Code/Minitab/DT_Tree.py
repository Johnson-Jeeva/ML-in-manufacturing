from mtbpy import mtbpy
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# ---------------------------
# Connect to Minitab
# ---------------------------
mtb = mtbpy.mtb_instance()

# ---------------------------
# Get command-line arguments
# ---------------------------
if len(sys.argv) < 3:
    mtb.add_message("Usage: pysc 'DecisionTree_model.py' <Target_Col> <Max_Col> [--depth N] [--minsplit N]")
    sys.exit(1)

target_col = sys.argv[1]      # Example: C1
max_col    = sys.argv[2]      # Example: C220

# Default parameters
max_depth = 4
min_samples_split = 2

# Parse optional parameters
for i, arg in enumerate(sys.argv):
    if arg == "--depth" and i + 1 < len(sys.argv):
        max_depth = int(sys.argv[i + 1])
    elif arg == "--minsplit" and i + 1 < len(sys.argv):
        min_samples_split = int(sys.argv[i + 1])

# ---------------------------
# Build list of columns
# ---------------------------
max_idx = int(max_col[1:])
all_columns = [f"C{i}" for i in range(1, max_idx + 1)]

# Remove target column from predictors
predictor_cols = [col for col in all_columns if col != target_col]

# ---------------------------
# Get target column data
# ---------------------------
y = np.array(mtb.get_column(target_col))

# ---------------------------
# Get predictor columns data
# ---------------------------
X_data = []
valid_cols = []
for col in predictor_cols:
    try:
        values = mtb.get_column(col)
        X_data.append(values)
        valid_cols.append(col)
    except:
        mtb.add_message(f"Warning: Column {col} not found. Skipping.")

# Prepare data matrix
X_np = np.array(X_data).T
X_df = pd.DataFrame(X_np, columns=valid_cols).dropna()

# Remove only zero-variance features
X_df = X_df.loc[:, X_df.var() > 1e-12]
valid_cols = X_df.columns.tolist()
y = y[:len(X_df)]

# ---------------------------
# Split data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

# Feature scaling (optional for Decision Trees, but keeps consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Train Decision Tree model
# ---------------------------
model = DecisionTreeRegressor(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ---------------------------
# Metrics
# ---------------------------
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# ---------------------------
# Write results to Minitab Output pane
# ---------------------------
mtb.add_message("Model: Decision Tree Regressor")
mtb.add_message(f"Parameters: Depth={max_depth}, MinSplit={min_samples_split}")
mtb.add_message(f"Target column: {target_col}")
mtb.add_message(f"Predictors used: {len(valid_cols)}")
mtb.add_message(f"Train R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
mtb.add_message(f"Test  R²: {test_r2:.4f} | RMSE: {test_rmse:.4f}")

# ---------------------------
# Plot 1: Actual vs Predicted (Training Set)
# ---------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_train_pred, y_train, color='black', s=10)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='black', linewidth=1)
plt.title("Training -  Actual vs Predicted (Decision Tree)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("DT_train_actual_vs_predicted.png", dpi=300)
mtb.add_image("DT_train_actual_vs_predicted.png")

# ---------------------------
# Plot 2: Actual vs Predicted (Test Set)
# ---------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test_pred, y_test, color='black', s=10)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linewidth=1)
plt.title("Test -  Actual vs Predicted (Decision Tree)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("DT_test_actual_vs_predicted.png", dpi=300)
mtb.add_image("DT_test_actual_vs_predicted.png")

# ---------------------------
# Plot 3: Residual Analysis (Test Set)
# ---------------------------
residuals = y_test - y_test_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_test_pred, residuals, color='black', s=10)
plt.axhline(0, color='black', linewidth=1)
plt.title("Residual Analysis (Decision Tree)")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("DT_residual_plot.png", dpi=300)
mtb.add_image("DT_residual_plot.png")

# ---------------------------
# Plot 4: Tree Visualization
# ---------------------------
plt.figure(figsize=(24, 12))
plot_tree(
    model,
    feature_names=valid_cols,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=max_depth
)
plt.title("Decision Tree Splits")
plt.savefig("DT_tree_split.png", dpi=300)
mtb.add_image("DT_tree_split.png")