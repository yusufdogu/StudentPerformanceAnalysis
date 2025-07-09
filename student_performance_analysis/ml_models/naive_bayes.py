import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
mat_19_s = pd.read_csv('mat_scalevar_binary_19_sutun.csv')
mat_33_s = pd.read_csv('mat_scalevar_binary_33_sutun.csv')
datasets = {'mat_19_sutun': mat_19_s, 'mat_33_sutun': mat_33_s}

# Naive Bayes Hyperparameter Grid (Opsiyonel, Naive Bayes genelde varsayılan değerlerle iyidir)
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

for name, df in datasets.items():
    print("=" * 60)
    print(f"🔮 Evaluating Dataset with Naive Bayes: {name}")
    print("=" * 60)

    X = df.drop(columns=['Basari'])
    y = df['Basari']

    # Train-test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Grid Search (on training set)
    grid_search = GridSearchCV(GaussianNB(), param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_full, y_train_full)
    best_params = grid_search.best_params_
    print(f"✅ Best Parameters: {best_params}")

    # Manual Stratified K-Fold to find best-performing fold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    best_acc = 0
    best_split = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model = GaussianNB(**best_params)
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_split = (train_idx, val_idx)

    print(f"\n🥇 Best Fold Accuracy: {best_acc * 100:.2f}%")

    # Retrain on best fold
    X_best_train, X_best_val = X_train_full.iloc[best_split[0]], X_train_full.iloc[best_split[1]]
    y_best_train, y_best_val = y_train_full.iloc[best_split[0]], y_train_full.iloc[best_split[1]]

    final_model = GaussianNB(**best_params)
    final_model.fit(X_best_train, y_best_train)
    y_test_pred = final_model.predict(X_test)

    # Evaluation Metrics
    print(f"🎯 Test Accuracy on Best Split: {accuracy_score(y_test, y_test_pred) * 100:.2f}%")
    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Naive Bayes Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
