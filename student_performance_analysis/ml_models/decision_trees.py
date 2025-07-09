import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# CSV dosyalarından veri setlerini yükle
ohe_scaled18_binary_target_mat_df = pd.read_csv('ohe_scaled18_binary_target_mat_df.csv')
ohe_scaled33_binary_target_mat_df = pd.read_csv('ohe_scaled33_binary_target_mat_df.csv')
ohe_scaled20_binary_target_por_df = pd.read_csv('ohe_scaled20_binary_target_por_df.csv')
ohe_scaled33_binary_target_por_df = pd.read_csv('ohe_scaled33_binary_target_por_df.csv')

# Hepsini bir sözlükte toplama
dfs = {
    'ohe_scaled18_binary_target_mat_df': ohe_scaled18_binary_target_mat_df,
    'ohe_scaled33_binary_target_mat_df': ohe_scaled33_binary_target_mat_df,
    'ohe_scaled20_binary_target_por_df': ohe_scaled20_binary_target_por_df,
    'ohe_scaled33_binary_target_por_df': ohe_scaled33_binary_target_por_df
}


# Hiperparametreler için grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Her veri seti için işlem
for name, df in dfs.items():
    print("=" * 60)
    print(f"🌳 Decision Tree Evaluation: {name}")
    print("=" * 60)

    X = df.drop(columns=['Basari'])
    y = df['Basari']

    # Eğitim-test bölmesi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Grid Search
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"✅ Best Parameters: {best_params}")

    # En iyi model
    model = DecisionTreeClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Tahmin ve değerlendirme
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")
    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name} - Decision Tree")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
