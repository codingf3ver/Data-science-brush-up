import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import psycopg2
from pymongo import MongoClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Config ---------------------- #
MODEL_PATH = "model.pkl"
ACCURACY_PATH = "accuracy.txt"
DB_TYPE = "mongo"  # options: "postgres" or "mongo"

# PostgreSQL config
POSTGRES_CONFIG = {
    "dbname": "demo",
    "user": "postgres",
    "password": "postgres@7003",
    "host": "localhost",
    "port": 5432,
}

# MongoDB config
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "ml_logs"
MONGO_COLLECTION = "predictions"

# ---------------------- Database Functions ---------------------- #
def log_prediction(features, prediction_label):
    if DB_TYPE == "postgres":
        log_prediction_postgres(features, prediction_label)
    elif DB_TYPE == "mongo":
        log_prediction_mongo(features, prediction_label)

def fetch_predictions():
    if DB_TYPE == "postgres":
        return fetch_predictions_postgres()
    elif DB_TYPE == "mongo":
        return fetch_predictions_mongo()
    return pd.DataFrame()

# --- PostgreSQL backend --- #
def log_prediction_postgres(features, prediction_label):
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                feature1 FLOAT,
                feature2 FLOAT,
                feature3 FLOAT,
                feature4 FLOAT,
                prediction TEXT
            );
        """)
        cur.execute("""
            INSERT INTO predictions (timestamp, feature1, feature2, feature3, feature4, prediction)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (datetime.now(), *features, prediction_label))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"[PostgreSQL] Error logging prediction: {e}")

def fetch_predictions_postgres():
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        df = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"[PostgreSQL] Error fetching predictions: {e}")
        return pd.DataFrame()

# --- MongoDB backend --- #
def log_prediction_mongo(features, prediction_label):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        if MONGO_COLLECTION not in db.list_collection_names():
            db.create_collection(MONGO_COLLECTION)
        collection = db[MONGO_COLLECTION]
        collection.insert_one({
            "timestamp": datetime.now(),
            "features": features,
            "prediction": prediction_label
        })
        client.close()
    except Exception as e:
        st.error(f"[MongoDB] Error logging prediction: {e}")

def fetch_predictions_mongo():
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        if MONGO_COLLECTION not in db.list_collection_names():
            return pd.DataFrame()
        collection = db[MONGO_COLLECTION]
        records = list(collection.find({}, {"_id": 0}))
        client.close()
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"[MongoDB] Error fetching predictions: {e}")
        return pd.DataFrame()


# ---------------------- Model Functions ---------------------- #
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names

def train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(clf, MODEL_PATH)
    with open(ACCURACY_PATH, "w") as f:
        f.write(str(acc))

    return clf, acc, y_test, y_pred

def load_model():
    if os.path.exists(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)
        with open(ACCURACY_PATH, "r") as f:
            acc = float(f.read())
        return clf, acc
    else:
        df, _ = load_data()
        return train_model(df)

# ---------------------- Streamlit UI ---------------------- #
def main():
    st.set_page_config("Iris Classifier with DB Logging", layout="wide")
    st.title("🌸 Iris Classifier with PostgreSQL/MongoDB Logging")

    df, target_names = load_data()

    if st.sidebar.button("Train Model from Scratch"):
        clf, acc, _, _ = train_model(df)
        st.sidebar.success("Model retrained.")
    else:
        clf, acc = load_model()

    st.sidebar.metric("Model Accuracy", f"{acc:.2f}")
    st.sidebar.info(f"Using database: `{DB_TYPE}`")

    if acc < 0.9:
        if st.button("⚠️ Retrain Model (Low Accuracy)"):
            clf, acc, _, _ = train_model(df)
            st.success("Model retrained.")

    st.subheader("🔍 Make a Prediction")
    input_data = []
    for col in df.columns[:-1]:
        val = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_data.append(val)

    if st.button("Predict"):
        prediction = clf.predict([input_data])[0]
        label = target_names[prediction]
        st.success(f"🌼 Predicted Class: {label}")
        log_prediction(input_data, label)

    with st.expander("📄 View Prediction Logs"):
        pred_df = fetch_predictions()
        if not pred_df.empty:
            st.dataframe(pred_df)
        else:
            st.info("No logs yet.")

    with st.expander("📉 Confusion Matrix"):
        _, _, y_test, y_pred = train_model(df)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
