# Logistic Regression — Notebooks

This repository contains multiple notebooks exploring logistic regression for binary and multiclass tasks, handling imbalanced data, and evaluating models using ROC curves.

Files
- `LogisticRegression.ipynb` — Baseline logistic regression workflow.
- `LogisticRegressionWithImbData.ipynb` — Techniques for imbalanced datasets (resampling, class weights, evaluation).
- `OneVsRest.ipynb` — Multiclass classification using One-vs-Rest strategy with logistic regression.
- `ROC.ipynb` — ROC curve plotting and AUC calculation for binary/multiclass tasks.

What the notebooks do — step-by-step
1. Dataset loading & EDA
   - Load dataset, inspect features and target distribution.
   - Visualize imbalance and feature-target relationships.

2. Preprocessing
   - Clean data, encode categorical variables, scale numerical features if needed.

3. Train/test split
   - Use `train_test_split` and cross-validation for robust evaluation.

4. Modeling
   - `LogisticRegression.ipynb`: Train a baseline `sklearn.linear_model.LogisticRegression` model, inspect coefficients, predict and evaluate.
   - `LogisticRegressionWithImbData.ipynb`: Handle class imbalance using techniques such as:
     - `class_weight='balanced'`
     - Resampling: oversampling (SMOTE) or undersampling
     - Evaluate using precision/recall/F1 instead of accuracy when imbalance exists.
   - `OneVsRest.ipynb`: Convert multiclass problem to multiple binary classifiers (One-vs-Rest) and evaluate per-class performance.
   - `ROC.ipynb`: Compute ROC curves and AUC for binary classifiers and show macro/micro-averaged ROC for multiclass.

5. Evaluation & visualization
   - Use confusion matrices, classification reports, precision/recall, and ROC/AUC visualizations to compare models.

Quick start
1. Install dependencies:
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter imbalanced-learn

2. Run:
   jupyter notebook
   # open the notebooks and run cells

Notes & tips
- For imbalanced datasets prefer precision/recall and PR curves; ROC can be misleading when classes are very imbalanced.
- Use `class_weight='balanced'` for a quick adjustment, and SMOTE/oversampling for stronger corrections.
- Inspect model coefficients to understand feature influence — logistic regression is interpretable and useful for explaining predictions.
