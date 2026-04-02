# Machine Learning in Manufacturing – Model Testing

This repository provides ready-to-use Python scripts for building, training, and evaluating machine learning models on manufacturing datasets. It is designed for **JMP** and **Minitab** users, enabling quick experimentation with predictive models for applications such as **process optimization, quality control, and performance forecasting**.

---

## Purpose

- **Comparison:** Train and compare multiple ML models on structured manufacturing datasets.
- **Evaluation:** Assess model performance using real or simulated industrial data.
- **Workflow:** Practice feature selection, validation set creation, and performance visualization.
- **Optimization:** Enable parameter tuning for improved accuracy and document results effectively.

---

## Supported Models

### 1. Classic & Advanced Regression
- **Linear:** Multiple Linear Regression (MLR), Ridge, Lasso, Elastic Net.
- **Tree-Based:** Decision Tree, Random Forest, XGBoost Regressors.
- **Deep Learning:** Artificial Neural Networks (ANN).
- **Hybrid:** Stacking models (ANN + Random Forest).

### 2. Anomaly Detection
- **Unsupervised:** Autoencoder (Neural Network) for detecting process outliers using Reconstruction Error.

Each model script provides:
- Automated Train/Validation data splitting.
- Predicted vs. Actual plots and Residual analysis.
- Accuracy metrics ($R^2$, $RMSE$, $MAE$).
- Feature Importance ranking or coefficient visualization.

---

## Dataset

The primary dataset for testing is:
- **Exercise 15 Regression Trees Strength2.xlsx**

*Note: Ensure this dataset is active in JMP or loaded into the Minitab worksheet before execution.*

---

## How to Use

### 1. Environment Setup
1. Install **Python 3.10+**.
2. Open your terminal and install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost shap mtbpy
   ```

### 2. Running in JSL (JMP)
1. Open **JMP** and load your data table.
2. Open and run the desired `.jsl` script (e.g., `Anomaly_Detection.jsl` or `RandomForest.jsl`).
3. Use the built-in UI to select your **Target** and **Predictor** columns.
4. View the generated results table and interactive Graph Builder plots.

### 3. Running in Minitab (PYSC)
Use the **PYSC** command in the Minitab command line. 

**General Analysis:**
`pysc "C:\Path\To\Script.py" "Target_Column" "Last_Column"`

**Anomaly Detection:**
`pysc "C:\Path\To\AnomalyDetection.py" "Timestamp_Column" "Last_Column"`

**Custom Prediction:**
`pysc "C:\Path\To\Script.py" "predict" "Target" "Last_Col" "Model_ID" "Value1,Value2..."`

---

## Model Identifiers (for Predict Command)

| Category | ID | Model |
| :--- | :--- | :--- |
| **Linear** | `mlr`, `ridge`, `lasso`, `elnet` | Linear/Regularized |
| **Trees** | `dt`, `rf`, `xgb` | Decision Tree, Random Forest, XGBoost |
| **Advanced** | `ann`, `hybrid` | Neural Net, ANN+RF Hybrid |
| **Anomaly** | `autoencoder` | Outlier Detection |

---

## Outputs & Interpretation

- **Regression:** Outputs Predicted Values, Residuals, and Feature Importance to identify which process variables (e.g., Temperature, Pressure) drive quality.
- **Anomaly Detection:** Generates a **Reconstruction Error Score** (higher = more unusual) and an **Anomaly Flag** (1 = Outlier, 0 = Normal).
