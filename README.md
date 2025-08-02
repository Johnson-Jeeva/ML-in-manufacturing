# Machine Learning in Manufacturing – Model Testing

This repository provides ready-to-use Python scripts for building, training and evaluating machine learning models on manufacturing datasets. It is designed for **JMP** and **Minitab** users, enabling quick experimentation with predictive models for applications such as **process optimization, quality control and performance forecasting**.

---

## Purpose

- Train and compare multiple machine learning models on structured manufacturing datasets.  
- Evaluate model performance using real or simulated industrial data.  
- Practice feature selection, validation set creation, and performance visualization.  
- Enable quick parameter tuning for improved accuracy and document results effectively.  

---

## Supported Models

- Decision Tree Regressor  
- Random Forest Regressor  
- XGBoost Regressor  
- Ridge Regression  
- Lasso Regression  

Each model script provides:
- Train/Test data split  
- Predicted vs Actual plots  
- Residual analysis  
- Model accuracy metrics (R², RMSE)  
- Feature importance or coefficient visualization (where applicable)  

---

## Dataset

The dataset used for model testing is:

- Exercise 15 Regression Trees Strength2.xlsx

Note: This dataset should be opened in **JMP** or loaded into **Minitab** before running the scripts.

---

## How to Use

### 1. Python Setup
1. Install **Python 3.10 or 3.11** from [python.org](https://www.python.org/).  
2. Run the `LibrariesInstall.py` script to install all required dependencies:

---

### 2. Running Scripts in JMP
1. Open **JMP** and load the dataset.  
2. Open the desired `.jsl` script:
- Decision_Tree
- RandomForest
- XG_Boost_Regressor
- Ridge
- Lasso
3. Run the script to:
- Train the model and see the evaluation metrics
- Generate a results table (Predicted, Actual, Residual, Data Split)  
- Save relevant plots (PNG) for model evaluation  

---

### 3. Running Scripts in Minitab
1. Load the dataset in **Minitab**.  
2. Use the **PYSC** command to run Python scripts. Example:
```bash
PYSC "C:\Path\To\Script.py" "<Target_Column>" "<Max_Column>" [parameters]
