import streamlit as st
import pandas as pd
import joblib

# Load your pre-trained pipeline and model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

st.title("Predict from Uploaded Data")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df)

    # Preprocess data
    try:
        df= df[df["Delay_Days"] > 0]
        df.drop(columns = ["Unnamed: 0","Delay_Days", "Customers_Location", "Repayment_Status"], inplace =True)

        processed_data = preprocessor.transform(df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # Make predictions
    predictions = model.predict(processed_data)
    y_prob = model.predict_proba(processed_data)[:,-1]

    # Add predictions to DataFrame
    df["Prediction"] = predictions
    df["Probability"] = y_prob
    st.write("### Predictions")
    st.dataframe(df)
#
    # Option to download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )
