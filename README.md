# Customer Churn Prediction

<div align="center">

### 🔵 Machine Learning Classification Project

**An end-to-end Customer Churn Prediction system with an interactive Streamlit dashboard**

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)

</div>

---

## 🌟 Overview

**Customer Churn Prediction** is an end-to-end machine learning classification project designed to predict whether a customer is likely to churn based on customer service and usage information.

The project goes beyond training a single model by covering the complete machine learning workflow:

```text
📊 Data
   ↓
🔎 Exploratory Data Analysis
   ↓
🧹 Data Preprocessing
   ↓
🤖 Model Training
   ↓
📈 Model Evaluation
   ↓
⚙️ Hyperparameter Tuning
   ↓
🎯 Threshold Analysis
   ↓
🔍 Feature Importance
   ↓
🖥️ Streamlit Dashboard
```

The final application allows users to interact with the trained model and explore the underlying dataset, model results, business-oriented observations, and prediction history.

---

## 🎯 Project Objective

The main objective is to build a machine learning solution that can classify customers according to the `Churn` target.

The project also investigates multiple classification models and evaluation metrics to understand their performance and behavior.

---

# 📊 Dataset

The project uses:

```text
churn_dataset.csv
```

The dataset contains **2,666 rows and 20 columns**.

The target variable is:

```text
Churn
```

The target contains two classes:

* `True` — Customer churn
* `False` — Customer does not churn

### Dataset Features

The customer information used by the project includes:

| Feature                | Type        |
| ---------------------- | ----------- |
| State                  | Categorical |
| Account length         | Numerical   |
| Area code              | Numerical   |
| International plan     | Categorical |
| Voice mail plan        | Categorical |
| Number vmail messages  | Numerical   |
| Total day minutes      | Numerical   |
| Total day calls        | Numerical   |
| Total day charge       | Numerical   |
| Total eve minutes      | Numerical   |
| Total eve calls        | Numerical   |
| Total eve charge       | Numerical   |
| Total night minutes    | Numerical   |
| Total night calls      | Numerical   |
| Total night charge     | Numerical   |
| Total intl minutes     | Numerical   |
| Total intl calls       | Numerical   |
| Total intl charge      | Numerical   |
| Customer service calls | Numerical   |
| Churn                  | Target      |

### Churn Distribution

The dataset contains:

| Class   | Customers |
| ------- | --------: |
| `False` |     2,278 |
| `True`  |       388 |

This means the target classes are imbalanced, which is why the project also investigates class balancing and metrics such as Precision, Recall, and F1 Score.

---

# 🔎 Exploratory Data Analysis

The notebook performs several stages of EDA to understand the dataset before modeling.

### Data Overview

The project examines:

* Dataset shape
* Column names
* Data types
* Statistical summary
* Number of unique values

### Data Quality

The notebook checks:

* Missing values
* Duplicate rows
* Unique values for each column

### Churn Distribution

The target distribution is visualized using a count plot to understand the number of churned and non-churned customers.

### Numerical Analysis

Box plots are used to investigate churn in relation to:

* `Total day minutes`
* `Customer service calls`
* `Total intl calls`
* `Account length`

### Categorical Analysis

The notebook investigates:

* `International plan`
* `Voice mail plan`

against the churn target.

### Correlation Analysis

A numerical correlation matrix and heatmap are generated to examine relationships between numerical variables.

---

# 🧹 Data Preprocessing

Categorical variables are transformed using one-hot encoding:

```python
df = pd.get_dummies(df, drop_first=True)
```

The target is separated from the feature set:

```python
X = df.drop("Churn", axis=1)
y = df["Churn"]
```

The dataset is split using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Feature Scaling

`StandardScaler` is used for the Logistic Regression models.

The tree-based models use the encoded features without applying the scaler.

```text
                    Encoded Data
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
       Logistic Regression     Tree-Based Models
              │                     │
       StandardScaler          No Scaling
              │                     │
              ↓                     ↓
          Prediction             Prediction
```

This distinction is important because the saved scaler is **not part of the Random Forest prediction pipeline**.

---

# 🤖 Machine Learning Models

The project evaluates multiple classification approaches.

## 1. Logistic Regression

A baseline Logistic Regression model trained using scaled features.

## 2. Balanced Logistic Regression

A Logistic Regression model using:

```python
class_weight="balanced"
```

to account for the imbalanced target classes.

## 3. Random Forest

A Random Forest classifier trained using the encoded, unscaled feature data.

## 4. Gradient Boosting

A Gradient Boosting classifier trained and evaluated as another tree-based approach.

## 5. Tuned Random Forest

Random Forest hyperparameters are optimized using `GridSearchCV`.

The parameter search includes:

```python
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

---

# 📈 Model Evaluation

The project evaluates classification performance using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

Model comparison results are stored in:

```text
model_comparison_results.csv
```

The comparison file is also used by the Streamlit dashboard.

---

# 🔄 Cross-Validation

The project uses **Stratified 5-Fold Cross-Validation**:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The following metrics are evaluated during cross-validation:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
```

Stratification helps maintain the class distribution across the folds.

---

# ⚙️ Hyperparameter Tuning

The Random Forest model is further optimized using `GridSearchCV`.

The search explores combinations of:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

The purpose is to investigate whether changing the Random Forest configuration can improve its validation performance.

---

# 🎯 Threshold Analysis

The project does not only rely on the default classification threshold.

Different probability thresholds are evaluated to investigate their effect on classification performance.

The results are stored in:

```text
threshold_results.csv
```

The Streamlit **Model Lab** also provides a visual threshold-analysis section.

This allows the relationship between the classification threshold and evaluation metrics to be examined directly from the project results.

---

# 🔍 Feature Importance

Feature importance is calculated for the Random Forest model.

The results are saved in:

```text
feature_importance.csv
```

The Streamlit application uses these results to visualize the relative importance of the model features.

> Feature importance describes how the trained model uses the available features. It does not by itself establish a causal relationship between a feature and customer churn.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application branded as **ChurnAI**.

The dashboard provides multiple sections covering both the machine learning model and the surrounding analysis.

---

## 🏠 Home

The Home page provides an overview of:

* The Customer Churn Prediction project
* Dataset information
* Prediction functionality
* Machine learning workflow
* Key project statistics

---

## 🎯 Predict

The prediction interface allows users to enter customer information and generate an individual churn prediction.

The page provides:

* Customer input fields
* Churn prediction
* Churn probability
* Risk level
* Prediction-related signals
* Feature importance information
* Prediction report generation
* Save-to-history functionality

---

## 📊 Explore Data

The Explore Data page provides interactive dataset analysis, including:

* Dataset statistics
* Data quality information
* Churn distribution
* Categorical churn analysis
* Numerical distributions
* Raw dataset preview
* Dataset download

---

## 💼 Business Insights

The Business Insights page presents descriptive churn analysis across customer characteristics.

The application includes analysis related to factors such as:

* International plan
* Voice mail plan
* Customer service calls
* State

These observations are descriptive and are not presented as causal conclusions.

---

## 🧪 Model Lab

The Model Lab provides a dedicated view of the machine learning workflow.

It includes:

* Model comparison
* Evaluation metrics
* ROC analysis
* Confusion matrix
* Feature importance
* Threshold analysis
* Model details
* Cross-validation information

---

## 🗂️ Prediction History

The Prediction History page allows users to work with previously saved predictions.

Available functionality includes:

* View saved predictions
* Prediction statistics
* Refresh history
* Download prediction history as CSV
* Clear prediction history

The application stores prediction history in:

```text
prediction_history.csv
```

---

## 📘 About Project

The About Project page presents information about:

* Project objective
* Machine learning workflow
* Technology stack
* Project architecture
* Project assets
* Developer information

---

# 🔮 Prediction Workflow

The application's prediction flow is:

```text
👤 Customer Information
          ↓
🧹 Input Preparation
          ↓
🔢 Feature Encoding & Alignment
          ↓
🌲 Random Forest Model
          ↓
📊 Churn Probability
          ↓
🎯 Churn Prediction
          ↓
🚦 Risk Level
          ↓
🗂️ Optional History
```

### Step-by-step

1. Enter customer information.
2. Submit the prediction.
3. The application prepares and encodes the input.
4. Input features are aligned with the trained model's expected features.
5. The Random Forest model generates the prediction.
6. The application displays the churn probability.
7. A corresponding risk level is displayed.
8. The prediction can optionally be saved to history.
9. Saved predictions can later be viewed or downloaded.

---

# 📦 Model Artifacts

The project contains several generated artifacts:

| File                           | Description                                           |
| ------------------------------ | ----------------------------------------------------- |
| `customer_churn_model.pkl`     | Saved trained model used by the application           |
| `scaler.pkl`                   | Saved scaler used in the Logistic Regression workflow |
| `model_comparison_results.csv` | Model evaluation comparison results                   |
| `threshold_results.csv`        | Probability threshold analysis results                |
| `feature_importance.csv`       | Random Forest feature importance results              |

---

# 🛠️ Technologies

The project is built using:

| Technology      | Usage                          |
| --------------- | ------------------------------ |
| 🐍 Python       | Core programming language      |
| 🐼 Pandas       | Data manipulation and analysis |
| 🔢 NumPy        | Numerical operations           |
| 📊 Matplotlib   | Data visualization             |
| 📈 Seaborn      | Statistical visualization      |
| 🤖 Scikit-learn | Machine learning               |
| 🖥️ Streamlit   | Interactive dashboard          |
| 📦 Joblib       | Model/artifact handling        |

---

# 📁 Project Structure

```text
Z1-Customer Churn Prediction/
│
├── app.py
│
├── Customer_Churn_Prediction_Final.ipynb
│
├── churn_dataset.csv
│
├── customer_churn_model.pkl
├── scaler.pkl
│
├── model_comparison_results.csv
├── threshold_results.csv
├── feature_importance.csv
│
├── prediction_history.csv
│
├── requirements.txt
│
└── README.md
```

> `prediction_history.csv` is used by the Streamlit application for saved prediction history.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
```

## 2. Open the Project

```bash
cd Z1-Customer-Churn-Prediction
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

```bash
streamlit run app.py
```

After starting Streamlit, open the local URL provided in the terminal.

---

# 💻 Using the Application

Once the application is running:

```text
Home
  ↓
Predict
  ↓
Enter Customer Data
  ↓
Generate Prediction
  ↓
Review Probability & Risk
  ↓
Save Prediction
  ↓
Prediction History
```

The remaining sections can be used to explore the dataset, business observations, model evaluation, feature importance, and threshold analysis.

---

# 📄 Project Notebook

The complete machine learning workflow is documented in:

```text
Customer_Churn_Prediction_Final.ipynb
```

The notebook covers:

```text
EDA
 ↓
Data Quality
 ↓
Preprocessing
 ↓
Logistic Regression
 ↓
Balanced Logistic Regression
 ↓
Random Forest
 ↓
Gradient Boosting
 ↓
Model Comparison
 ↓
Cross-Validation
 ↓
Random Forest Tuning
 ↓
Threshold Analysis
 ↓
Feature Importance
 ↓
Model Saving
```

---

# 📥 Generated Outputs

The project generates and uses several analysis outputs:

### Model Comparison

```text
model_comparison_results.csv
```

### Threshold Analysis

```text
threshold_results.csv
```

### Feature Importance

```text
feature_importance.csv
```

### Saved Model

```text
customer_churn_model.pkl
```

### Saved Scaler

```text
scaler.pkl
```

### Prediction History

```text
prediction_history.csv
```

---

# 🔐 Important Implementation Note

The project contains two different preprocessing paths:

### Logistic Regression

```text
Categorical Encoding
        ↓
Train/Test Split
        ↓
StandardScaler
        ↓
Logistic Regression
```

### Random Forest

```text
Categorical Encoding
        ↓
Train/Test Split
        ↓
Random Forest
```

Therefore, the `scaler.pkl` file should not be applied to the Random Forest model used by the Streamlit application.

---

# 🔮 Future Improvements

Possible future development areas include:

* Persistent cloud-based prediction history
* User authentication
* Automated model retraining
* Model monitoring
* Additional explainability techniques
* Production database integration
* Expanded deployment infrastructure

These are potential future improvements rather than currently implemented features.

---

# 👩‍💻 Author

<div align="center">

## Noura Maher Elamin

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0A66C2?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/nouramaherelamin/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/nouramaherelamin)

</div>

---

<div align="center">

### 📊 Customer Churn Prediction

**Machine Learning • Data Analysis • Model Evaluation • Streamlit**

<br>

© 2026 **Noura Maher Elamin**
**All Rights Reserved**

</div>
