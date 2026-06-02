import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

REPO_OWNER = "ArcVielLouvent"
REPO_NAME = "Experiment_SML_Armand-Al-Farizy"
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

df = pd.read_csv('../preprocessing/dataset_preprocessing/data_clean.csv')
features = df.select_dtypes(include='number').columns.tolist()
target = features.pop()

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Advanced_Tuning_Experiment")

with mlflow.start_run():
    print("Sedang melakukan Tuning...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    print("Menyimpan Artefak Tambahan...")

    feat_imp = pd.DataFrame(
        {'Feature': X.columns, 'Importance': best_model.feature_importances_})
    feat_imp.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(best_model, "random_forest_model")

    os.remove("feature_importance.csv")
    os.remove("confusion_matrix.png")

    print("Cek dashboard DagsHub.")
