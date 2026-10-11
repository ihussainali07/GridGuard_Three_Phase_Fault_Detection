# ⚡ GridGuard: Three-Phase Transmission-Line Fault Detection

**A Machine Learning Project | AI4003 Applied Machine Learning — Assignment 2**

GridGuard is a machine learning-based project that analyzes three-phase electrical current and voltage measurements to classify transmission-line operating conditions as normal or faulty. It combines a manually implemented Logistic Regression model with Linear and RBF-kernel Support Vector Machines (SVMs) to investigate classification performance.

The project includes data preparation, exploratory data analysis (EDA), feature engineering, model training, validation, held-out test evaluation, and a Streamlit dashboard for interactive model inference.

The main objective is to explore how machine learning can support electrical fault detection by identifying patterns in three-phase current and voltage measurements.

---

## 🖥️ Application UI Preview

**Streamlit Dashboard**

![GridGuard_Three_Phase_Fault_Detection](interface.png)

**How to add your screenshot:**
1. Create a folder named `docs` in the project root.
2. Inside `docs`, create a folder named `images`.
3. Save your Streamlit dashboard screenshot as `gridguard-dashboard.png` inside `docs/images`.
4. Commit and push the image to GitHub.

Once the screenshot is uploaded, it will appear in this section.

**Dashboard screenshot:** `docs/images/gridguard-dashboard.png`

---

## 🎯 Project Objectives

- ⚡ Analyze three-phase electrical current and voltage measurements.
- 📊 Perform exploratory data analysis and visualize feature distributions.
- 🧮 Implement Logistic Regression manually.
- 🔍 Compare Set R and Set Q feature configurations.
- 🤖 Train and evaluate Linear and RBF-kernel SVM classifiers.
- 📈 Compare validation accuracy and held-out test performance.
- 🧪 Analyze confusion matrices, false alarms, and missed faults.
- 🌐 Provide an interactive Streamlit dashboard for model inference.

## ✨ Key Features

- **Data Preparation:** Loads and prepares the supplied transmission-line dataset.
- **Exploratory Data Analysis:** Generates current time-series plots, scatter plots, and histograms.
- **Manual Logistic Regression:** Implements model training and cost-history tracking.
- **Feature Engineering:** Evaluates the Set R and Set Q configurations used in the assignment.
- **SVM Classification:** Evaluates Linear and RBF-kernel Support Vector Machines.
- **Model Evaluation:** Reports accuracy, true negatives, false positives, false negatives, and true positives.
- **Visualization:** Saves EDA plots, Logistic Regression cost curves, and SVM validation figures.
- **Interactive Dashboard:** Uses Streamlit to provide a user interface for predictions.
- **Saved Model Artifacts:** Stores trained models for reuse during inference.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical calculations and array operations |
| Pandas | Dataset loading and manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | SVM classification and evaluation utilities |
| Streamlit | Interactive web dashboard |
| Pickle | Saved model serialization |
| Git and GitHub | Version control and project hosting |

## 📂 Project Structure

The repository is organized into data, application, source-code, and results directories.

```text
GridGuard_Three_Phase_Fault_Detection/
│
├── app/
│   ├── __pycache__/
│   └── app.py
│
├── data/
│   └── detect_dataset.csv
│
├── results/
│   ├── figures/
│   │   ├── current_timeseries.png
│   │   ├── ia_histograms.png
│   │   ├── ia_ib_scatter.png
│   │   ├── logistic_cost_history_set_Q.png
│   │   ├── logistic_cost_history_set_R.png
│   │   └── svm_validation_accuracy_vs_C.png
│   │
│   ├── tables/
│   │   ├── table_1_data_preparation.csv
│   │   ├── table_2_logistic_regression.csv
│   │   ├── table_3_svm_validation_results.csv
│   │   ├── table_4_test_results.csv
│   │   ├── table_5_set_q_weights.csv
│   │   └── test_results.csv
│   │
│   ├── analysis.md
│   ├── logistic_Q_costs.npy
│   ├── logistic_Q.pkl
│   ├── logistic_R_costs.npy
│   ├── logistic_R.pkl
│   ├── svm_linear.pkl
│   └── svm_rbf.pkl
│
├── src/
│   ├── __pycache__/
│   ├── eda.py
│   ├── load_data.py
│   ├── logistic.py
│   ├── metrics.py
│   ├── prepare.py
│   ├── test_models.py
│   ├── train_logistic.py
│   └── train_svm.py
│
├── .gitignore
├── README.md
└── testing
```

## 🧠 Machine Learning Models

### 1. Logistic Regression

Logistic Regression is implemented manually to explore binary classification, cost calculation, and model training.

Two feature configurations are evaluated:

- **Set R:** Baseline feature configuration.
- **Set Q:** Expanded feature configuration that includes squared features.

The comparison investigates how feature engineering affects classification performance.

### 2. Linear Support Vector Machine

The Linear SVM is evaluated as a baseline classifier. Different values of the regularization parameter \(C\) are compared during validation.

### 3. RBF Support Vector Machine

The RBF SVM uses a nonlinear kernel to learn more flexible decision boundaries. Hyperparameters are evaluated using validation data before the final comparison on the held-out test set.

## 📊 Test Results

The following table summarizes the recorded test results for the four evaluated models.

| Model | Test Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|
| Logistic Regression — Set R | 72.64% | 1,305 | 0 | 657 | 439 |
| Logistic Regression — Set Q | 97.88% | 1,299 | 6 | 45 | 1,051 |
| Linear SVM | 73.59% | 1,305 | 0 | 634 | 462 |
| RBF SVM | 99.58% | 1,301 | 4 | 6 | 1,090 |

**Metric definitions:**

- **TN — True Negatives:** Normal cases correctly classified as normal.
- **FP — False Positives:** Normal cases incorrectly classified as faults.
- **FN — False Negatives:** Fault cases incorrectly classified as normal.
- **TP — True Positives:** Fault cases correctly classified as faults.

### 🔎 Results Interpretation

- The RBF SVM achieved **99.58% test accuracy**, with four false alarms and six missed faults.
- Logistic Regression with Set Q achieved **97.88% test accuracy**, compared with 72.64% for Set R.
- The Linear SVM achieved 73.59% test accuracy on the evaluated test split.
- The results illustrate the importance of feature engineering and the choice of classification model.

These findings apply to the evaluated dataset and test split. Additional testing on independent datasets and realistic electrical operating conditions would be required before considering practical deployment.

## 📈 Exploratory Data Analysis and Visualizations

The `results/figures/` directory contains the visualizations generated during the analysis.

| Figure | Description |
|---|---|
| `current_timeseries.png` | Time-series visualization of current measurements |
| `ia_histograms.png` | Distribution of phase A current measurements |
| `ia_ib_scatter.png` | Scatter plot comparing phase A and phase B currents |
| `logistic_cost_history_set_Q.png` | Logistic Regression cost history for Set Q |
| `logistic_cost_history_set_R.png` | Logistic Regression cost history for Set R |
| `svm_validation_accuracy_vs_C.png` | SVM validation accuracy comparison across values of \(C\) |

These figures support the investigation of feature distributions, model convergence, and validation performance.

## 📋 Results Tables

The `results/tables/` directory stores the generated CSV reports.

| File | Purpose |
|---|---|
| `table_1_data_preparation.csv` | Data preparation results |
| `table_2_logistic_regression.csv` | Logistic Regression results |
| `table_3_svm_validation_results.csv` | SVM validation results |
| `table_4_test_results.csv` | Held-out test performance |
| `table_5_set_q_weights.csv` | Learned Set Q coefficients |
| `test_results.csv` | Additional test results table |

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your computer
- Visual Studio Code or another Python editor
- The project dataset
- Required Python packages

### 1. Open the project directory

Open the `GridGuard_Three_Phase_Fault_Detection` folder in Visual Studio Code.

Open a terminal from the project root.

### 2. Create a virtual environment

If the existing environment is unavailable, create a new one:

```powershell
python -m venv .venv
```

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scikit-learn streamlit joblib
```

## ▶️ Running the Project

Run the following commands **from the project root and in the order shown**.

### Step 1: Prepare the data

```powershell
.\.venv\Scripts\python.exe src\prepare.py
```

### Step 2: Train Logistic Regression

```powershell
.\.venv\Scripts\python.exe src\train_logistic.py
```

### Step 3: Train the SVM models

```powershell
.\.venv\Scripts\python.exe src\train_svm.py
```

### Step 4: Evaluate the models

```powershell
.\.venv\Scripts\python.exe src\test_models.py
```

The training scripts generate or update the relevant model artifacts and results according to their implementation.

## 🌐 Launch the Streamlit Dashboard

Run the following command from the project root:

```powershell
.\.venv\Scripts\streamlit.exe run app\app.py
```

Alternatively:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

Once the application starts, open the local URL displayed in your terminal. The default address is usually:

`http://localhost:8501`

The application is designed to use the preprocessing and feature ordering required by the saved models.

## 📥 Input Measurements and Units

The model uses six electrical input features.

| Feature | Description | Required input convention |
|---|---|---|
| `Ia` | Phase A current | Raw units used in the dataset |
| `Ib` | Phase B current | Raw units used in the dataset |
| `Ic` | Phase C current | Raw units used in the dataset |
| `Va` | Phase A voltage | Per-unit (p.u.) |
| `Vb` | Phase B voltage | Per-unit (p.u.) |
| `Vc` | Phase C voltage | Per-unit (p.u.) |

**Important notes:**

- The supplied CSV does not specify the physical unit of current. Do not assume that current values are amperes or kiloamperes.
- Voltage measurements should be entered as per-unit values, as specified by the assignment.
- The application does not convert units before inference.
- Use input values consistent with the dataset and the model's training preprocessing.

## ⚠️ Limitations

- Results depend on the supplied dataset and the selected data split.
- High test accuracy does not guarantee performance on different transmission lines or operating conditions.
- The physical current units are not identified in the supplied CSV.
- The dashboard is an educational demonstration, not a certified electrical protection system.
- Independent validation and domain-specific testing are needed before any real-world use.

## 🚀 Future Improvements

- Add further datasets representing diverse electrical operating conditions.
- Expand model evaluation with precision, recall, F1-score, and ROC-AUC.
- Improve the dashboard with interactive charts and prediction summaries.
- Add model comparison visualizations directly to the dashboard.
- Investigate additional feature-engineering methods and classifiers.
- Validate models on independent data.

## 🎓 Academic Information

**Course:** AI4003 — Applied Machine Learning  
**Assignment:** Assignment 2 — Transmission-Line Fault Detection

This project documents the implementation and evaluation of machine learning methods for a binary transmission-line fault-detection task.

## 📚 References

1. **Scikit-learn — Support Vector Classification (SVC)**  
   https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

2. **Streamlit — Create an App**  
   https://docs.streamlit.io/get-started/tutorials/create-an-app

3. **NumPy — Random Sampling Documentation**  
   https://numpy.org/doc/stable/reference/random/index.html

4. **Dataset:** Add the original dataset source and required attribution here.

5. **Assignment:** AI4003 Applied Machine Learning Assignment 2 handout.

---

## 👨‍💻 Author

**HUSSAIN ALI**

- GitHub: `https://github.com/YOUR-USERNAME`
- Repository: `GridGuard_Three_Phase_Fault_Detection`

---

<p align="center">
  ⚡ <strong>GridGuard</strong> — Applying Machine Learning to Three-Phase Fault Detection.
</p>
