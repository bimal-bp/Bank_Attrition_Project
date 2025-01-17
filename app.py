import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define columns expected by the model
expected_columns = [
    'Customer_Age', 'Credit_Limit', 'Total_Transactions_Count',
    'Total_Transaction_Amount', 'Inactive_Months_12_Months',
    'Transaction_Count_Change_Q4_Q1', 'Total_Products_Used',
    'Average_Credit_Utilization', 'Customer_Contacts_12_Months',
    'Transaction_Amount_Change_Q4_Q1', 'College', 'Doctorate', 'Graduate',
    'High School', 'Post-Graduate', 'Uneducated', '$120K +', '$40K - $60K',
    '$60K - $80K', '$80K - $120K', 'Less than $40K'
]

# Helper function to create input data
def create_input_data(age, credit_limit, transactions_count, transaction_amount, inactive_months,
                      transaction_count_change, products_used, credit_utilization, contacts,
                      transaction_amount_change, education, income):
    # Initialize the input data
    input_data = {col: 0 for col in expected_columns}
    input_data.update({
        'Customer_Age': age,
        'Credit_Limit': credit_limit,
        'Total_Transactions_Count': transactions_count,
        'Total_Transaction_Amount': transaction_amount,
        'Inactive_Months_12_Months': inactive_months,
        'Transaction_Count_Change_Q4_Q1': transaction_count_change,
        'Total_Products_Used': products_used,
        'Average_Credit_Utilization': credit_utilization,
        'Customer_Contacts_12_Months': contacts,
        'Transaction_Amount_Change_Q4_Q1': transaction_amount_change
    })

    # Set the appropriate education column
    if education in input_data:
        input_data[education] = 1

    # Set the appropriate income column
    if income in input_data:
        input_data[income] = 1

    return pd.DataFrame([input_data])

# Streamlit app
st.title("Bank Customer Churn Prediction")

# Input fields
age = st.slider("Customer Age", min_value=18, max_value=100, value=35)
credit_limit = st.number_input("Credit Limit", min_value=0.0, step=100.0, value=5000.0)
transactions_count = st.number_input("Total Transactions Count", min_value=0, step=1, value=50)
transaction_amount = st.number_input("Total Transaction Amount", min_value=0.0, step=10.0, value=3000.0)
inactive_months = st.slider("Inactive Months in Last 12 Months", min_value=0, max_value=12, value=3)
transaction_count_change = st.number_input("Transaction Count Change (Q4 vs Q1)", min_value=-5.0, max_value=5.0, step=0.1, value=0.5)
products_used = st.slider("Total Products Used", min_value=1, max_value=10, value=3)
credit_utilization = st.slider("Average Credit Utilization Ratio", min_value=0.0, max_value=1.0, step=0.01, value=0.3)
contacts = st.slider("Customer Contacts in Last 12 Months", min_value=0, max_value=12, value=3)
transaction_amount_change = st.number_input("Transaction Amount Change (Q4 vs Q1)", min_value=-5.0, max_value=5.0, step=0.1, value=0.5)

# Dropdowns for education and income
education = st.selectbox(
    "Education Level",
    ["College", "Doctorate", "Graduate", "High School", "Post-Graduate", "Uneducated"]
)
income = st.selectbox(
    "Income Bracket",
    ["$120K +", "$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K"]
)

# Prediction
if st.button("Predict"):
    input_data = create_input_data(age, credit_limit, transactions_count, transaction_amount, inactive_months,
                                   transaction_count_change, products_used, credit_utilization, contacts,
                                   transaction_amount_change, education, income)

    # Ensure columns are in the correct order
    input_data = input_data[expected_columns]

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(scaled_input)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    st.success(f"Prediction: {result}")
