
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import os

if __name__ == "__main__":
    input_data_path = "/opt/ml/input/data/train/diabetes.csv"
    data = pd.read_csv(input_data_path)
    X = data.drop(columns=['target'])
    y = data['target']

    model = LinearRegression()
    model.fit(X, y)

    model_dir = os.environ["SM_MODEL_DIR"]
    joblib.dump(model, f"{model_dir}/model.joblib")
    
