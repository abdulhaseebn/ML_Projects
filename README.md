
# Breast Cancer Prediction using Machine Learning

Breast cancer is one of the most common and life-threatening cancers affecting women globally. Early detection plays a vital role in improving survival rates and reducing unnecessary treatment. This project uses machine learning techniques to classify whether a tumor is benign or malignant based on features extracted from digitized images of breast cell nuclei.

# Objective

To build and compare multiple machine learning models that can classify tumors as benign or malignant using the Wisconsin Breast Cancer Diagnostic dataset, and identify the best-performing model.

# Dataset
Source: UCI Machine Learning Repository

Features: 30 numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses.

Target: 0 = Benign, 1 = Malignant

Link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

# Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Colab (for development)

# Data Preprocessing

- Removed irrelevant columns like `id`.
- Converted diagnosis labels (M, B) to binary numeric (1, 0).
- Checked for null values and data imbalance.
- Applied feature scaling (StandardScaler).

# Exploratory Data Analysis

Visualizations included:
- Correlation Heatmap
- Class distribution
- Pair plots of selected features

# Models Trained

| Model                  | Description |
|------------------------|-------------|
|Logistic Regression    |Simple linear model for binary classification |
|Random Forest         |Ensemble tree-based classifier |
|K-Nearest Neighbors    |Distance-based classifier with hyperparameter tuning |
|Support Vector Machine |Kernel-based classifier (linear & RBF) |
|XGBoost            | Extreme gradient boosting |
|Naive Bayes            | Probabilistic classifier based on Bayes’ Theorem |
|Dense Neural Network (MLP)   | Multi-layer Perceptron using `sklearn.neural_network` |

# Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

# Models Performance Summary

| Model  | Accuracy	| Precision	|Recall	|F1-Score
|--------|----------| ----------| ------| -------|
|Logistic Regression |96%|97%|96%|96%|
|Random Forest |97%|97%|97%|97%|
|KNN |94%|94%|94%|94%|
|SVM |96%|97%|96%|96%|
|Naive Bayes |92%|92%|92%|92%|
|XGBoost |97%|97%|97%|97%|
|**Neural Network**	|**98%**|98%|98%|98%|

# Hyperparameter Tuning

Used `GridSearchCV` & `RandomizedSearchCV` to find best hyperparameters for:
  - KNN: `n_neighbors`, `weights`, `metric`
  - XGBClassifier: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `gamma`
  - Random Forest: `n_estimators`, `max_depth`, `max_features`

# Conclusion

- The **Neural Network** achieved the highest performance (98% accuracy, 0.98 F1).
- Random Forest and XGBoost also performed strongly with excellent recall and precision.

⭐ **Feel free to fork this repo or suggest improvements!**

