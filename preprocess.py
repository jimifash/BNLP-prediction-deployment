import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pipeline import Preprocessor
import joblib

# Load data
df = pd.read_csv("Cleaned_new (1).csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

categorical_cols = [
    'Loyalty_Program_Status', 'Gender', 'Segment', 'Customer_Duration',
    'Age_Group', 'Repayment_Status', 'Customers_Location', 'Currency',
    'Order Status', 'Payment_Method', 'ProductCategory', 'ProductBrand',
    'ProductCategory2', 'Stock Status', 'Product Status',
    'Order Channel', 'Payment_Mode'
]
