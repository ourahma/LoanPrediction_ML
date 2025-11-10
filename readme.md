# Comparative Analysis of Machine Learning Models for Loan Classification

A web-based application to predict loan approval status using multiple machine learning models. Users can input applicant details via a **Streamlit** UI and get predictions from several models simultaneously. This project demonstrates the end-to-end process of preprocessing data, training models, an making the predictions.

---

## Features

* Input form for applicant details:

  * Gender, Marital Status, Dependents, Education, Self-Employed
  * Applicant Income, Coapplicant Income, Loan Amount, Loan Term
  * Credit History, Property Area
* Predict loan approval (`Loan_Status`) using multiple pre-trained ML models:

  * Random Forest
  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Support Vector Classifier (SVC)
  * Gradient Boosting classifier.
  * Stacking/Ensemble model combining all base models
  * Voting Hard/Soft.
* Handles missing data and categorical encoding automatically.
* Displays predictions from all models at once for comparison.

---

## Machine Learning Models Explained

1. **Random Forest (RF)**

   * An ensemble learning method that builds multiple decision trees and merges their predictions.
   * Pros: Handles non-linear data well, reduces overfitting by averaging multiple trees.

2. **Logistic Regression (LR)**

   * A linear model used for binary classification.
   * Outputs probabilities that an applicant will be approved for a loan.
   * Pros: Simple, interpretable, effective for linearly separable data.

3. **K-Nearest Neighbors (KNN)**

   * Predicts the class of a data point based on the majority class among its k nearest neighbors.
   * Pros: Non-parametric, simple to implement, adapts to complex decision boundaries.

4. **Support Vector Classifier (SVC)**

   * Finds the hyperplane that best separates classes in high-dimensional space.
   * Pros: Effective in high-dimensional spaces, robust to overfitting with proper kernel and regularization.

5. **Stacking / Ensemble Model**

   * Combines predictions from multiple base models (RF, LR, KNN, SVC) using a meta-classifier.
   * Pros: Often more accurate than individual models as it leverages strengths of each base model.

6. **Gradient boosting classifier :**
    * Builds an ensemble of weak prediction models, typically decision trees, in a stage-wise fashion to optimize predictive performance.
    * Pros: It delivers state-of-the-art predictive accuracy and naturally handles feature interactions and mixed data types.
    * Cons: It is computationally expensive, sensitive to outliers, and requires extensive hyperparameter tuning to prevent overfitting.

7. **Voting Hard Model**

    * Combines base model predictions using majority vote.
    * Pros: Simple to implement, robust to outliers in predictions.
    * Cons: Does not consider prediction confidence, can be biased if base models are imbalanced.

8. **Voting Soft Model**

    * Combines base model predictions by averaging probabilities.
    * Pros: Considers confidence of each model, often more accurate than hard voting.
    * Cons: Requires probability outputs from all base models, more sensitive to poorly calibrated models.
---

## Installation

1. Clone the repository:

```bash
git clone git@github.com:ourahma/LoanPrediction_ML.git
cd lLoanPrediction_ML
```

2. Create a virtual environment and activate it:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/macOS
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

* Fill in the form with applicant details.
* Click **Predict**.
* Predictions from all models will be displayed.

---

## Folder Structure

```
loan-prediction-app/
│
├── dataset/          # Raw or processed dataset (ignored in Git)
├── models/           # Saved ML models (joblib files)
├── processors/       # Preprocessing scripts
├── .venv/            # Virtual environment (ignored in Git)
├── app.py            # Streamlit app
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Preprocessing

* **Missing Value Handling**: Automatically fills missing numeric values with median and categorical values with the most frequent.
* **Encoding**: Categorical features are label-encoded.
* **Scaling**: Numeric features are standardized to ensure models like KNN and SVC perform optimally.

All preprocessing steps are implemented in the `processors/` folder and applied before predictions.

---

## ScreenCast


https://github.com/user-attachments/assets/1f43fcb7-787b-44be-a70f-4416ce5dd9a6




---

## Notes

* `.gitignore` excludes dataset files, virtual environment, and saved models.
* Models can be retrained by updating scripts in `processors/` and saving new versions in `models/`.

---

## Author:

- **OURAHMA Maroua.**


