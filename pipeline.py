from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import joblib
import os

class Preprocessor:
    def __init__(self, categorical_cols, drop_original=True, encoders_path="encoders", scaler_path="scaler.pkl"):
        self.categorical_cols = categorical_cols
        self.drop_original = drop_original
        self.encoders_path = encoders_path
        self.scaler_path = scaler_path
        self.encoders = {}
        self.scaler = None
        self.numeric_cols = []

    def fit(self, df):
        """Fit encoders and scaler dynamically based on data."""
        os.makedirs(self.encoders_path, exist_ok=True)

        # Fit and save LabelEncoders
        for col in self.categorical_cols:
            if col in df.columns:
                encoder = LabelEncoder()
                encoder.fit(df[col])
                self.encoders[col] = encoder
                joblib.dump(encoder, os.path.join(self.encoders_path, f"{col}_encoder.pkl"))

        # Identify numeric columns
        self.numeric_cols = df.drop(columns=[c for c in self.categorical_cols if c in df.columns]) \
                             .select_dtypes(include="number").columns.tolist()

        # Fit and save MinMaxScaler
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[self.numeric_cols])
        joblib.dump(self.scaler, self.scaler_path)

    def transform(self, df):
        """Transform data using fitted encoders and scaler."""
        # Load encoders if not already loaded
        if not self.encoders:
            for col in self.categorical_cols:
                encoder_file = os.path.join(self.encoders_path, f"{col}_encoder.pkl")
                if os.path.exists(encoder_file):
                    self.encoders[col] = joblib.load(encoder_file)

        if not self.scaler:
            self.scaler = joblib.load(self.scaler_path)

        # Encode categorical columns
        for col in self.categorical_cols:
            if col in df.columns:
                df[f"{col}_en"] = self.encoders[col].transform(df[col])

        # Drop Customers_Location entirely
        if "Customers_Location" in df.columns:
            df = df.drop(columns=["Customers_Location"])

        # Drop other categorical columns that exist (but keep Repayment_Status for y)
        drop_cols = [c for c in self.categorical_cols if c != "Repayment_Status" and c in df.columns]
        if self.drop_original and drop_cols:
            df = df.drop(columns=drop_cols)

        # Scale numeric columns
        existing_numeric_cols = [col for col in self.numeric_cols if col in df.columns]
        if existing_numeric_cols:
            df[existing_numeric_cols] = self.scaler.transform(df[existing_numeric_cols])

        return df

    def fit_transform(self, df):
        """Fit encoders & scaler, then transform the data."""
        self.fit(df)
        return self.transform(df)
