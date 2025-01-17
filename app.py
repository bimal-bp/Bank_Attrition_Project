import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
best_rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the list of columns (same as used for training)
columns = [
    "Customer_Age", "Credit_Limit", "Total_Transactions_Count", "Total_Transaction_Amount", 
    "Inactive_Months_12_Months", "Transaction_Count_Change_Q4_Q1", "Total_Products_Used", 
    "Average_Credit_Utilization", "Customer_Contacts_12_Months", "Transaction_Amount_Change_Q4_Q1", 
    "Months_as_Customer", "College", "Doctorate", "Graduate", "High School", "Post-Graduate", 
    "Uneducated", "$120K +", "$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K"
]

# Function to predict churn
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Ensure the column order is correct and add missing columns if necessary
    input_df = input_df[columns]
    
    # Add missing column (e.g., 'Months_as_Customer') with placeholder values
    if 'Months_as_Customer' not in input_df.columns:
        input_df['Months_as_Customer'] = [12]  # Placeholder value
    
    # Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # Predict using the loaded model
    prediction = best_rf_model.predict(scaled_input)
    
    # Return the prediction
    return "Churn" if prediction[0] == 1 else "No Churn"

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on various input features.")

# Create input fields for the user to enter data
customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
credit_limit = st.number_input("Credit Limit", min_value=1000, max_value=50000, value=5000)
total_transactions_count = st.number_input("Total Transactions Count", min_value=0, value=50)
total_transaction_amount = st.number_input("Total Transaction Amount", min_value=0, value=3000)
inactive_months = st.number_input("Inactive Months in 12 Months", min_value=0, value=1)
transaction_count_change_q4_q1 = st.number_input("Transaction Count Change Q4 to Q1", min_value=0.0, value=0.5)
total_products_used = st.number_input("Total Products Used", min_value=1, value=2)
average_credit_utilization = st.number_input("Average Credit Utilization", min_value=0.0, max_value=1.0, value=0.3)
customer_contacts_12_months = st.number_input("Customer Contacts in 12 Months", min_value=0, value=2)
transaction_amount_change_q4_q1 = st.number_input("Transaction Amount Change Q4 to Q1", min_value=0.0, value=0.4)

# User-selected education level (one-hot encoding)
college = st.selectbox("Education: College", options=[0, 1], index=1)
doctorate = st.selectbox("Education: Doctorate", options=[0, 1], index=0)
graduate = st.selectbox("Education: Graduate", options=[0, 1], index=0)
high_school = st.selectbox("Education: High School", options=[0, 1], index=0)
post_graduate = st.selectbox("Education: Post-Graduate", options=[0, 1], index=0)
uneducated = st.selectbox("Education: Uneducated", options=[0, 1], index=0)

# Income bracket selection (one-hot encoding)
income_brackets = ["$120K +", "$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K"]
income_selected = st.selectbox("Income Bracket", options=income_brackets, index=4)

# Encode the income and education fields
income_data = {bracket: 1 if income_selected == bracket else 0 for bracket in income_brackets}
education_data = {
    "College": college, "Doctorate": doctorate, "Graduate": graduate, 
    "High School": high_school, "Post-Graduate": post_graduate, "Uneducated": uneducated
}

# Prepare input data
input_data = {
    "Customer_Age": customer_age,
    "Credit_Limit": credit_limit,
    "Total_Transactions_Count": total_transactions_count,
    "Total_Transaction_Amount": total_transaction_amount,
    "Inactive_Months_12_Months": inactive_months,
    "Transaction_Count_Change_Q4_Q1": transaction_count_change_q4_q1,
    "Total_Products_Used": total_products_used,
    "Average_Credit_Utilization": average_credit_utilization,
    "Customer_Contacts_12_Months": customer_contacts_12_months,
    "Transaction_Amount_Change_Q4_Q1": transaction_amount_change_q4_q1,
    **education_data,
    **income_data
}

# Predict churn based on the input data
if st.button("Predict Churn"):
    prediction = predict_churn(input_data)
    st.write(f"Prediction: The customer is likely to {prediction}")
