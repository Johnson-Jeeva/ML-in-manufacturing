from mtbpy import mtbpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ---------------------------
# Connect to Minitab
# ---------------------------
mtb = mtbpy.mtb_instance()

# ---------------------------
# Get command-line arguments
# ---------------------------
target_col = sys.argv[1]   # Example: C5
max_col    = sys.argv[2]   # Example: C20
max_depth = 6  # default
for i, arg in enumerate(sys.argv):
    if arg == "--maxdepth" and i+1 < len(sys.argv):
        max_depth = int(sys.argv[i+1])

# Build list of columns from C1 to Cn
max_idx = int(max_col[1:])  # Convert C20 -> 20
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
y = y[:len(X_df)]

# ---------------------------
# Split data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

# ---------------------------
# Train Random Forest model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=max_depth,
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
mtb.add_message("Model: Random Forest Regressor")
mtb.add_message(f"Target column: {target_col}")
mtb.add_message(f"Max Depth: {max_depth}")
mtb.add_message(f"Train R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
mtb.add_message(f"Test  R²: {test_r2:.4f} | RMSE: {test_rmse:.4f}")

# ---------------------------
# Plot 1: Actual vs Predicted (Training Set)
# ---------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_train_pred, y_train, color='black', s=10)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='black', linewidth=1)
plt.title("Training Set: Actual vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("RF_train_actual_vs_predicted.png", dpi=300)
mtb.add_image("RF_train_actual_vs_predicted.png")

# ---------------------------
# Plot 2: Actual vs Predicted (Test Set)
# ---------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test_pred, y_test, color='black', s=10)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linewidth=1)
plt.title("Test Set: Actual vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("RF_test_actual_vs_predicted.png", dpi=300)
mtb.add_image("RF_test_actual_vs_predicted.png")

# ---------------------------
# Plot 3: Tree Visualization (first tree)
# ---------------------------
plt.figure(figsize=(10, 6))
plot_tree(model.estimators_[0], feature_names=valid_cols, filled=False, max_depth=3, fontsize=6)
plt.title("Tree Visualization (Depth ≤ 3)")
plt.tight_layout()
plt.savefig("RF_tree_visualization.png", dpi=300)
mtb.add_image("RF_tree_visualization.png")

# ---------------------------
# Plot 4: Feature Importance (Top-N)
# ---------------------------
importance = model.feature_importances_
top_n = min(10, len(valid_cols))
sorted_idx = np.argsort(importance)[-top_n:]

plt.figure(figsize=(6, 5))
plt.barh(np.array(valid_cols)[sorted_idx], importance[sorted_idx], color='black')
plt.title(f"Top {top_n} Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("RF_feature_importance_topN.png", dpi=300)
mtb.add_image("RF_feature_importance_topN.png")

# ---------------------------
# Plot 5: Residual Analysis (Test Set)
# ---------------------------
residuals = y_test - y_test_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_test_pred, residuals, color='black', s=10)
plt.axhline(0, color='black', linewidth=1)
plt.title("Residual Analysis (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("RF_residual_analysis.png", dpi=300)
mtb.add_image("RF_residual_analysis.png")

# ---------------------------
# Plot 6: Permutation Importance (Top-N)
# ---------------------------
perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_sorted_idx = perm_result.importances_mean.argsort()[-top_n:]

plt.figure(figsize=(6, 4))
plt.barh(np.array(valid_cols)[perm_sorted_idx], perm_result.importances_mean[perm_sorted_idx], color='black')
plt.title(f"Top {top_n} Permutation Importance")
plt.xlabel("Mean Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("RF_permutation_importance_topN.png", dpi=300)
mtb.add_image("RF_permutation_importance_topN.png")