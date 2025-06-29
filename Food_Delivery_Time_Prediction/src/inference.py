# src/inference.py

import joblib
import pandas as pd
from src.data_preparation import prepare_data

# Load preprocessing pipeline and model
preprocessor = joblib.load("models/preprocessing_pipeline.pkl")
model = joblib.load("models/stacking_model.pkl")  # TransformedTargetRegressor

from src.data_preparation import prepare_inference_data

def predict_delivery_time(input_data: dict) -> float:
    input_df = pd.DataFrame([input_data])
    processed_df = prepare_inference_data(input_df)

    if processed_df.empty:
        raise ValueError("Processed data is empty. Possibly due to invalid or missing fields.")

    X_transformed = preprocessor.transform(processed_df)
    predicted_time = model.predict(X_transformed)[0]
    return round(predicted_time, 2)
