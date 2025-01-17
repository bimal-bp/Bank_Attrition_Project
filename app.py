import streamlit as st
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler


# Example of saving the scaler
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler to the training data

# Save the scaler to a file
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Example of saving the model
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Function to preprocess input data
def preprocess_input(input_df, calr):
    # Apply the preprocessor (e.g., scaler, encoder) on the input
    processed_data = calr.transform(input_df)
    return processed_data

# Update the predict function
def predict(data):
    # Preprocess the data using calr.pkl before making predictions
    processed_data = preprocess_input(data, calr)
    prediction = model.predict(processed_data)
    return prediction

# Create the Streamlit app layout
st.title('Customer Churn Prediction')

st.sidebar.header('Input Data')

# Define input fields based on your features
Customer_Age = st.sidebar.number_input('Customer Age', min_value=18, max_value=100, value=30)
Credit_Limit = st.sidebar.number_input('Credit Limit', min_value=0, value=5000)
Total_Transactions_Count = st.sidebar.number_input('Total Transactions Count', min_value=0, value=50)
Total_Transaction_Amount = st.sidebar.number_input('Total Transaction Amount', min_value=0, value=1000)
Inactive_Months_12_Months = st.sidebar.number_input('Inactive Months (12 Months)', min_value=0, value=3)
Transaction_Count_Change_Q4_Q1 = st.sidebar.number_input('Transaction Count Change Q4/Q1', value=0)
Total_Products_Used = st.sidebar.number_input('Total Products Used', min_value=1, value=3)
Average_Credit_Utilization = st.sidebar.number_input('Average Credit Utilization', min_value=0.0, max_value=1.0, value=0.5)
Customer_Contacts_12_Months = st.sidebar.number_input('Customer Contacts (12 Months)', min_value=0, value=1)
Transaction_Amount_Change_Q4_Q1 = st.sidebar.number_input('Transaction Amount Change Q4/Q1', value=0)
Months_as_Customer = st.sidebar.number_input('Months as Customer', min_value=0, value=12)

# For categorical features like education, you can use selectbox
education = st.sidebar.selectbox('Education', ['College', 'Doctorate', 'Graduate', 'High School', 'Post-Graduate', 'Uneducated'])

# For income categories
income = st.sidebar.selectbox('Income', ['$120K +', '$40K - $60K', '$60K - $80K', '$80K - $120K', 'Less than $40K'])

# Prepare input data
input_data = {
    'Customer_Age': [Customer_Age],
    'Credit_Limit': [Credit_Limit],
    'Total_Transactions_Count': [Total_Transactions_Count],
    'Total_Transaction_Amount': [Total_Transaction_Amount],
    'Inactive_Months_12_Months': [Inactive_Months_12_Months],
    'Transaction_Count_Change_Q4_Q1': [Transaction_Count_Change_Q4_Q1],
    'Total_Products_Used': [Total_Products_Used],
    'Average_Credit_Utilization': [Average_Credit_Utilization],
    'Customer_Contacts_12_Months': [Customer_Contacts_12_Months],
    'Transaction_Amount_Change_Q4_Q1': [Transaction_Amount_Change_Q4_Q1],
    'Months_as_Customer': [Months_as_Customer],
    'College': [1 if education == 'College' else 0],
    'Doctorate': [1 if education == 'Doctorate' else 0],
    'Graduate': [1 if education == 'Graduate' else 0],
    'High School': [1 if education == 'High School' else 0],
    'Post-Graduate': [1 if education == 'Post-Graduate' else 0],
    'Uneducated': [1 if education == 'Uneducated' else 0],
    '$120K +': [1 if income == '$120K +' else 0],
    '$40K - $60K': [1 if income == '$40K - $60K' else 0],
    '$60K - $80K': [1 if income == '$60K - $80K' else 0],
    '$80K - $120K': [1 if income == '$80K - $120K' else 0],
    'Less than $40K': [1 if income == 'Less than $40K' else 0],
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Predict the result when button is clicked
if st.sidebar.button('Predict'):
    prediction = predict(input_df)
    st.write(f'Predicted Churn: {prediction[0]}')

# Run the app
if __name__ == '__main__':
    st.write('Use the sidebar to input the data and predict customer churn.')
