#  Predicting Secondary School Student Performance Using Data Mining Methods

This project focuses on predicting academic performance of secondary school students based on real-world datasets collected from two public schools in Portugal. The data includes not only grades but also demographic, social, and family-related factors that may impact student success.

##  Project Objective

The main goal of this study is to analyze the factors affecting student performance and to build machine learning models that can accurately predict final grades. By identifying key variables and evaluating different classification algorithms, this project aims to support early interventions and improve educational outcomes.

##  Dataset Overview

-  Origin: Two Portuguese secondary schools
-  Samples: 649 (Portuguese) + 395 (Math) students
-  Features: 33 attributes including demographics, family background, study habits, and lifestyle
-  Targets: G1, G2, G3 (Grades for 1st period, 2nd period, and final exam)

No missing values exist in the dataset. Both numerical and categorical variables are included, with appropriate preprocessing applied (e.g., One-Hot Encoding, MinMax Scaling).

##  Methodology

1. **Data Preprocessing**
   - Handling categorical variables
   - Scaling features using MinMaxScaler
   - Reducing features based on correlation analysis

2. **Feature Engineering**
   - Removal of highly correlated targets (G1, G2)
   - Averaging grades into a single target variable
   - Selecting variables with significant impact on final grade

3. **Modeling**
   - Classification task only
   - Applied models:
     - Decision Trees
     - Random Forest
     - Support Vector Machines (SVM)
     - Artificial Neural Networks (ANN)
   - Hyperparameter tuning with GridSearch
   - Cross-validation using k-Fold strategy

4. **Evaluation Metrics**
   - Accuracy (%) over reduced and full feature sets

| Model          | Math (18 vars) | Math (33 vars) | Portuguese (20 vars) | Portuguese (33 vars) |
|----------------|----------------|----------------|------------------------|------------------------|
| Random Forest  | 73.42%         | 74.68%         | 84.50%                 | 84.50%                 |
| Decision Tree  | 69.62%         | 68.35%         | 78.46%                 | 78.46%                 |
| SVM            | 78.48%         | 70.89%         | 81.40%                 | 81.40%                 |
| Neural Network | 72.15%         | 75.95%         | 81.40%                 | 81.54%                 |

## 💡 Key Findings

- Grades from earlier periods (G1 and G2) are highly correlated with final grade (G3).
- Other important predictors include:
  - Number of past class failures
  - Number of absences
  - Mother's and father's education level
  - Weekly study time
- Feature reduction helped achieve similar or better results with fewer variables.

##  Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook

##  Future Suggestions

- Incorporate psychological and emotional variables for richer insights.
- Apply time-series analysis to monitor student trends.
- Consider including teacher characteristics and course difficulty.

##  Project Structure

