from mtbpy import mtbpy
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os

# ---------------------------
# Connect to Minitab
# ---------------------------
mtb = mtbpy.mtb_instance()

# ---------------------------
# Setup Plots folder
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "Plots")
os.makedirs(plots_dir, exist_ok=True)

# ---------------------------
# Parse arguments
# ---------------------------
args = sys.argv[1:]
predict_mode = args[0].lower() == "predict"

if predict_mode:
    _, target_col, max_col, model_name, input_str = args
    input_values = [float(x.strip()) for x in input_str.split(",")]
else:
    target_col, max_col = args

# ---------------------------
# Build predictor columns
# ---------------------------
max_idx = int(max_col[1:])
predictor_cols = (
    [f"C{i}" for i in range(1, max_idx + 1) if f"C{i}" != target_col]
    if not predict_mode
    else [f"C{i}" for i in range(1, len(input_values)+1)]
)

# ---------------------------
# Load data from Minitab
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

X_df = pd.DataFrame(np.array(X_data).T, columns=valid_cols).dropna()
final_features = X_df.columns.tolist()

# ---------------------------
# Load target column
# ---------------------------
try:
    y = np.array(mtb.get_column(target_col))[:len(X_df)]
except:
    mtb.add_message(f"Error: Target column {target_col} not found in Minitab.")
    sys.exit()

# ---------------------------
# Define models
# ---------------------------
models = {
    "mlr": LinearRegression(),
    "ridge": RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5),
    "lasso": LassoCV(alphas=np.logspace(-2, 3, 50), cv=5, max_iter=20000, tol=1e-4, selection='random'),
    "elnet": ElasticNetCV(alphas=np.logspace(-2, 3, 50), l1_ratio=0.5, cv=5, max_iter=20000, tol=1e-4)
}

scaler = StandardScaler()

# ---------------------------
# Predict mode (single row)
# ---------------------------
if predict_mode:
    model_name_lower = model_name.lower()
    if model_name_lower not in models:
        mtb.add_message(f"Model {model_name} not recognized. Choose from: mlr, ridge, lasso, elnet.")
        sys.exit()

    # Train on full data
    X_scaled = scaler.fit_transform(X_df)
    model = models[model_name_lower]
    model.fit(X_scaled, y)

    # Convert input row to DataFrame with same columns
    input_df = pd.DataFrame([input_values], columns=final_features)
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]

    # Output summary only
    mtb.add_message(f"Predicted value: {pred:.4f} based on {model_name_lower.upper()}")
    sys.exit()

# ---------------------------
# Default mode (full analysis)
# ---------------------------
# Data cleaning
X_df = X_df.loc[:, X_df.var() > 1e-6]
corr_matrix = X_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [column for column in upper.columns if any(upper[column] > 0.98)]
if drop_cols:
    X_df = X_df.drop(columns=drop_cols)
    mtb.add_message(f"Dropped {len(drop_cols)} highly correlated columns.")
y = y[:len(X_df)]
final_features = X_df.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------
# Compute manager-friendly summary table
# ---------------------------
summary_list = []
model_preds = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    r2_test = r2_score(y_test, y_test_pred)
    mad = np.mean(np.abs(y_test - y_test_pred))
    summary_list.append({
        "Model": name.replace("_"," ").title(),
        "Test R-squared (%)": round(r2_test*100,2),
        "Mean Absolute Deviation": round(mad,4)
    })
    model_preds[name] = model  # Save trained model

# Identify best model
best_idx = np.argmax([s['Test R-squared (%)'] for s in summary_list])
for i, s in enumerate(summary_list):
    if i == best_idx:
        s['Model'] += " *"

summary_df = pd.DataFrame(summary_list)
summary_df = summary_df.sort_values("Test R-squared (%)", ascending=False)
mtb.add_message("Overall Model Comparison (Best Model *)")
mtb.add_message(summary_df.to_string(index=False))

# ---------------------------
# Generate plots and detailed metrics per model
# ---------------------------
for name, model in model_preds.items():
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred  = model.predict(X_test_scaled)

    # Show detailed metrics
    alpha_val = getattr(model, "alpha_", None)
    mtb.add_message(f"\nModel: {name.upper()}")
    mtb.add_message(f"Target column: {target_col}")
    mtb.add_message(f"Predictors used: {len(final_features)}")
    mtb.add_message(f"Train R²: {r2_score(y_train, y_train_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
    mtb.add_message(f"Test  R²: {r2_score(y_test, y_test_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
    if alpha_val is not None:
        mtb.add_message(f"Alpha: {alpha_val:.6f}")

    # Feature importance plot (Top 10)
    if hasattr(model, "coef_"):
        coef = model.coef_
        abs_coef_idx = np.argsort(np.abs(coef))[::-1]
        top_n = 10 if len(coef) > 10 else len(coef)
        top_idx = abs_coef_idx[:top_n]

        plt.figure(figsize=(7,5))
        plt.barh([final_features[i] for i in top_idx], coef[top_idx], color='black')
        plt.xlabel("Coefficient Value")
        plt.title(f"{name.upper()} Top Feature Importance")
        plt.tight_layout()
        fname = os.path.join(plots_dir, f"{name}_feature_importance.png")
        plt.savefig(fname, dpi=300)
        mtb.add_image(fname)
        plt.close()

    # Actual vs Predicted plots
    for data, y_true, y_pred, label in [("train", y_train, y_train_pred, "Training"), ("test", y_test, y_test_pred, "Test")]:
        plt.figure(figsize=(6,5))
        plt.scatter(y_pred, y_true, color='black', s=10)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='black', linewidth=1)
        plt.title(f"{name.upper()} - {label} Actual vs Predicted")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        fname = os.path.join(plots_dir, f"{name}_{data}_actual_vs_predicted.png")
        plt.savefig(fname, dpi=300)
        mtb.add_image(fname)
        plt.close()

    # Residuals plot
    residuals = y_test - y_test_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_test_pred, residuals, color='black', s=10)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"{name.upper()} Residual Analysis (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"{name}_residual_analysis.png")
    plt.savefig(fname, dpi=300)
    mtb.add_image(fname)
    plt.close()