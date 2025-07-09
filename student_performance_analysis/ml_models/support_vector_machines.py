import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
ohe_scaled18_binary_target_mat_df = pd.read_csv('ohe_scaled18_binary_target_mat_df.csv')
ohe_scaled33_binary_target_mat_df = pd.read_csv('ohe_scaled33_binary_target_mat_df.csv')
ohe_scaled20_binary_target_por_df = pd.read_csv('ohe_scaled20_binary_target_por_df.csv')
ohe_scaled33_binary_target_por_df = pd.read_csv('ohe_scaled33_binary_target_por_df.csv')

dfs = {
    'ohe_scaled18_binary_target_mat_df': ohe_scaled18_binary_target_mat_df,
    'ohe_scaled33_binary_target_mat_df': ohe_scaled33_binary_target_mat_df,
    'ohe_scaled20_binary_target_por_df': ohe_scaled20_binary_target_por_df,
    'ohe_scaled33_binary_target_por_df': ohe_scaled33_binary_target_por_df
}

# SVM için GridSearch parametreleri
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Her dataset için işlemler
for name, df in dfs.items():
    print("=" * 60)
    print(f"📊 Evaluating Dataset with SVM + GridSearch: {name}")
    print("=" * 60)

    X = df.drop(columns=['Basari'])
    y = df['Basari']

    best_acc = 0
    best_split = None
    best_model = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # GridSearchCV ile en iyi hiperparametreleri bul
        grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # En iyi modelle fold değerlendirmesi
        best_svm = grid_search.best_estimator_
        y_pred = best_svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Fold {fold + 1} Accuracy: {acc:.4f} | Best Params: {grid_search.best_params_}")

        if acc > best_acc:
            best_acc = acc
            best_split = (X_train, X_test, y_train, y_test)
            best_model = best_svm

    print(f"\n🥇 Best Fold Accuracy: {best_acc * 100:.2f}%")

    # Final modelle test değerlendirmesi
    X_train, X_test, y_train, y_test = best_split
    y_pred_test = best_model.predict(X_test)

    print(f"🎯 Final Test Accuracy (Best Fold): {accuracy_score(y_test, y_pred_test) * 100:.2f}%")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_test))

    conf_matrix = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
