from mtbpy import mtbpy
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
target_col = sys.argv[1]      # Example: C5
max_col    = sys.argv[2]      # Example: C220

# Build list of columns from C1 to Cn
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

# Prepare data matrix (drop rows with NaNs in X)
X_np = np.array(X_data).T
X_df = pd.DataFrame(X_np, columns=valid_cols).dropna()

# ---------------------------
# Data Cleaning
# ---------------------------

# 1) Remove zero/near-zero variance columns
X_df = X_df.loc[:, X_df.var() > 1e-6]

# 2) Remove highly correlated features (|r| > 0.98)
corr_matrix = X_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [column for column in upper.columns if any(upper[column] > 0.98)]
if drop_cols:
    X_df = X_df.drop(columns=drop_cols)
    mtb.add_message(f"Dropped {len(drop_cols)} highly correlated columns.")

# Update valid columns list
valid_cols = X_df.columns.tolist()

# Align target length to X_df (same approach as your LASSO script)
y = y[:len(X_df)]

# ---------------------------
# Train/Test Split + Scale
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ---------------------------
# Elastic Net with Cross-Validation
# ---------------------------
alphas = np.logspace(-2, 3, 80)   # 0.01 to 1000 (same scale as LASSO script)
l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # blend (1.0 == LASSO-like)

model = ElasticNetCV(
    l1_ratio=l1_grid,
    alphas=alphas,
    cv=5,
    random_state=42,
    max_iter=20000,
    tol=1e-4
)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

# ---------------------------
# Metrics
# ---------------------------
train_r2  = r2_score(y_train, y_train_pred)
test_r2   = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))

# ---------------------------
# Write results to Minitab Output pane
# ---------------------------
mtb.add_message(f"Model: Elastic Net (alpha={model.alpha_:.6f}, l1_ratio={model.l1_ratio_})")
mtb.add_message(f"Target column: {target_col}")
mtb.add_message(f"Predictors used: {len(valid_cols)}")
mtb.add_message(f"Train R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
mtb.add_message(f"Test  R²: {test_r2:.4f} | RMSE: {test_rmse:.4f}")

# ---------------------------
# Plot 1: Actual vs Predicted (Training Set)
# ---------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_train_pred, y_train, color='black', s=10)
mn, mx = min(y_train.min(), y_train_pred.min()), max(y_train.max(), y_train_pred.max())
plt.plot([mn, mx], [mn, mx], color='black', linewidth=1)
plt.title("Training - Actual vs Predicted (Elastic Net)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("ELNET_train_actual_vs_predicted.png", dpi=300)
mtb.add_image("ELNET_train_actual_vs_predicted.png")

# ---------------------------
# Plot 2: Actual vs Predicted (Test Set)
# ---------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test_pred, y_test, color='black', s=10)
mn, mx = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
plt.plot([mn, mx], [mn, mx], color='black', linewidth=1)
plt.title("Test - Actual vs Predicted (Elastic Net)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("ELNET_test_actual_vs_predicted.png", dpi=300)
mtb.add_image("ELNET_test_actual_vs_predicted.png")

# ---------------------------
# Plot 3: Top 25 Non-zero Coefficients
# ---------------------------
coef = model.coef_
nonzero_idx = np.where(coef != 0)[0]

if len(nonzero_idx) > 0:
    top_features = np.argsort(np.abs(coef[nonzero_idx]))[::-1][:25]
    plt.figure(figsize=(7, 5))
    plt.barh([valid_cols[nonzero_idx[i]] for i in top_features],
             coef[nonzero_idx][top_features], color='black')
    plt.title("Top 25 Elastic Net Non-zero Coefficients")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig("ELNET_top25_coefficients.png", dpi=300)
    mtb.add_image("ELNET_top25_coefficients.png")
else:
    mtb.add_message("No non-zero coefficients found with current alpha/l1_ratio.")

# ---------------------------
# Plot 4: Residual Analysis (Test Set)
# ---------------------------
residuals = y_test - y_test_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_test_pred, residuals, color='black', s=10)
plt.axhline(0, color='black', linewidth=1)
plt.title("Residual Analysis (Test Set) - Elastic Net")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("ELNET_residual_analysis.png", dpi=300)
mtb.add_image("ELNET_residual_analysis.png")