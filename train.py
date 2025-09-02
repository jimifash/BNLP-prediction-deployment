from preprocess import df_delay
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

# Separate features and target
X = df_delay.drop(columns=["Repayment_Status_en"])
y = df_delay["Repayment_Status_en"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(
    max_depth=8, random_state=42, n_estimators=1000, class_weight="balanced"
)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
