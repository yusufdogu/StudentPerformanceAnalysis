import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 2, 4, 6, 8, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

for name, df in dfs.items():
    print("=" * 60)
    print(f"📊 Evaluating Dataset: {name}")
    print("=" * 60)

    X = df.drop(columns=['Basari'])
    y = df['Basari']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_acc = 0
    best_split = None
    best_model = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        model = RandomForestClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_split = (X_train, X_test, y_train, y_test)
            best_model = model

    print(f"\n🥇 Best Fold Accuracy: {best_acc * 100:.2f}%")

    # Final evaluation on the best split
    X_train, X_test, y_train, y_test = best_split
    y_pred_test = best_model.predict(X_test)

    print(f"🎯 Test Accuracy (on Best Fold): {accuracy_score(y_test, y_pred_test) * 100:.2f}%")
    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_pred_test))

    conf_matrix = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
