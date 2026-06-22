# modelling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    df = pd.read_csv('titanic.csv')

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    le = LabelEncoder()
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train, X_test, y_test):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    acc_test = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)

    # =========================
    # 🔥 MLflow SECTION FIX
    # =========================
    mlflow.set_experiment("eksperimen_model")

    with mlflow.start_run():

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        mlflow.log_metric("accuracy_test", acc_test)
        mlflow.log_metric("accuracy_train", acc_train)

        mlflow.sklearn.log_model(model, "model")

    # save lokal juga
    joblib.dump(model, 'model.pkl')

    print("✅ Model trained successfully!")
    print(f"📊 Test Accuracy: {acc_test:.4f}")
    print(f"📊 Train Accuracy: {acc_train:.4f}")

    return model

if __name__ == "__main__":
    print("🚀 Starting Titanic Model Training...")

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    model = train_model(X_train, y_train, X_test, y_test)

    joblib.dump(scaler, 'scaler.pkl')

    print("✅ All files saved successfully!")
    print("📁 Files created:")
    print("   - model.pkl")
    print("   - scaler.pkl")
