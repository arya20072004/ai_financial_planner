import pandas as pd

def get_prediction_and_scaling(model, scaler, user_data, feature_columns):
    X_scaled = scaler.transform(pd.DataFrame([user_data])[feature_columns])
    return model.predict(X_scaled)[0]
