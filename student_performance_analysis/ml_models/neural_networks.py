import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt

# Verileri yükle
ohe_scaled18_binary_target_mat_df = pd.read_csv('ohe_scaled18_binary_target_mat_df.csv')
ohe_scaled33_binary_target_mat_df = pd.read_csv('ohe_scaled33_binary_target_mat_df.csv')
ohe_scaled20_binary_target_por_df = pd.read_csv('ohe_scaled20_binary_target_por_df.csv')
ohe_scaled33_binary_target_por_df = pd.read_csv('ohe_scaled33_binary_target_por_df.csv')

datasets = {
    'ohe_scaled18_binary_target_mat_df': ohe_scaled18_binary_target_mat_df,
    'ohe_scaled33_binary_target_mat_df': ohe_scaled33_binary_target_mat_df,
    'ohe_scaled20_binary_target_por_df': ohe_scaled20_binary_target_por_df,
    'ohe_scaled33_binary_target_por_df': ohe_scaled33_binary_target_por_df
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, df in datasets.items():
    print("=" * 60)
    print(f"🧠 Evaluating Dataset with Keras YSA: {name}")
    print("=" * 60)

    X = df.drop(columns=['Basari'])
    y = df['Basari']

    best_acc = 0
    best_split = None
    best_model = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Model tanımı
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        y_pred = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)

        print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_split = (X_train, X_test, y_train, y_test)
            best_model = model

    print(f"\n🥇 Best Fold Accuracy: {best_acc * 100:.2f}%")

    print(f"\n🥇 Best Fold Accuracy: {best_acc * 100:.2f}%")

    # Final evaluation on the best split
    X_train, X_test, y_train, y_test = best_split
    y_pred_test = (best_model.predict(X_test) > 0.5).astype(int)

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
