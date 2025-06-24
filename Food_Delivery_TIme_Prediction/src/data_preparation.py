# src/data_preparation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from src.data_clean_utils import perform_data_cleaning

def prepare_data(path: str):
    df = pd.read_csv(path)
    cleaned_df=perform_data_cleaning(df)
    
    columns_to_drop =  ['rider_id',
                        'restaurant_latitude',
                        'restaurant_longitude',
                        'delivery_latitude',
                        'delivery_longitude',
                        'order_date',
                        "order_time_hour",
                        "order_day",
                        "city_name",
                        "order_day_of_week",
                        "order_month"]
    
    cleaned_df.drop(columns=columns_to_drop, inplace=True)

    cleaned_df = cleaned_df.dropna()

    X = cleaned_df.drop(columns='time_taken')
    y = cleaned_df['time_taken']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pt = PowerTransformer()
    y_train_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_transformed = pt.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train, X_test, y_train_transformed, y_test_transformed, pt


def prepare_inference_data(df: pd.DataFrame):
    """
    Prepare a single data point for prediction.
    Applies cleaning and feature dropping — no target column used.
    """
    cleaned_df = perform_data_cleaning(df)

    columns_to_drop = [
        'rider_id', 'restaurant_latitude', 'restaurant_longitude',
        'delivery_latitude', 'delivery_longitude', 'order_date',
        'order_time_hour', 'order_day', 'city_name',
        'order_day_of_week', 'order_month'
    ]
    cleaned_df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    cleaned_df = cleaned_df.dropna()

    return cleaned_df
