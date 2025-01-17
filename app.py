import pandas as pd
import joblib

# Load the trained model and scaler
best_rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the saved scaler


# Original training columns (ensure the order matches exactly with the training data)
columns = [
    "Customer_Age", "Credit_Limit", "Total_Transactions_Count", "Total_Transaction_Amount", 
    "Inactive_Months_12_Months", "Transaction_Count_Change_Q4_Q1", "Total_Products_Used", 
    "Average_Credit_Utilization", "Customer_Contacts_12_Months", "Transaction_Amount_Change_Q4_Q1", 
    "Months_as_Customer", "College", "Doctorate", "Graduate", "High School", "Post-Graduate", 
    "Uneducated", "$120K +", "$40K - $60K", "$60K - $80K", "$80K - $120K", "Less than $40K"
]

# Input Data (Ensure the structure matches the training data exactly)
data = {
    "Customer_Age": [31, 40],
    "Credit_Limit": [3768.0, 2594.0],
    "Total_Transactions_Count": [71, 71],
    "Total_Transaction_Amount": [6639, 3872],
    "Inactive_Months_12_Months": [2, 1],
    "Transaction_Count_Change_Q4_Q1": [1.152, 0.511],
    "Total_Products_Used": [1, 6],
    "Average_Credit_Utilization": [0.000, 0.832],
    "Customer_Contacts_12_Months": [2, 1],
    "Transaction_Amount_Change_Q4_Q1": [0.799, 0.792],
    "College": [0, 0],
    "Doctorate": [0, 0],
    "Graduate": [1, 0],
    "High School": [0, 0],
    "Post-Graduate": [0, 0],
    "Uneducated": [0, 0],
    "$120K +": [0, 0],
    "$40K - $60K": [0, 0],
    "$60K - $80K": [0, 0],
    "$80K - $120K": [0, 0],
    "Less than $40K": [1, 1]
}

# Create a DataFrame for input data
input_df = pd.DataFrame(data)

# Add the missing column ('Months_as_Customer') with placeholder values (e.g., 12 and 24)
input_df['Months_as_Customer'] = [12, 24]  # Example placeholder values

# Ensure the column order is correct
input_df = input_df[columns]

# Check the columns after reordering
print("Reordered Input Data Columns:", input_df.columns)

# Scale the input data using the fitted scaler
scaled_input = scaler.transform(input_df)

# Predict using the loaded model
predictions = best_rf_model.predict(scaled_input)

# Output predictions
print("Predictions:", predictions)
