#!/usr/bin/env python
# coding: utf-8



# In[3]:


import streamlit as st
import numpy as np
import pickle  # Assuming the model is saved as a pickle file

# Define the category mapping
category_mapping = {
    "Gender": {"male": 1, "female": 0}, 
    "Married": {"yes": 1, "no": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"graduate": 0, "not graduate": 1},
    "Self_Employed": {"no": 0, "yes": 1},
    "Property_Area": {"semiurban": 0, "urban": 1, "rural": 2},  
}

# Function to preprocess text input
def preprocess_text_input(text_data):
    numerical_data = []
    keys = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    for key, value in zip(keys, text_data):
        if value in category_mapping.get(key, {}):  
            numerical_data.append(category_mapping[key][value])
        else:
            numerical_data.append(0)  
    return np.asarray(numerical_data)

# Load the trained model
# Assuming the model is saved in a file named 'loan_model.pkl'
with open('loan status prediction.sav', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Loan Prediction App")

# Collect text input
gender = st.selectbox("Gender", ["male", "female"])
married = st.selectbox("Married", ["yes", "no"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["graduate", "not graduate"])
self_employed = st.selectbox("Self Employed", ["yes", "no"])
property_area = st.selectbox("Property Area", ["semiurban", "urban", "rural"])

# Collect numerical input
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)

# Button for prediction
if st.button("Predict Loan Status"):
    # Preprocess text input
    text_input = [gender, married, dependents, education, self_employed, property_area]
    preprocessed_text_input = preprocess_text_input(text_input)

    # Combine categorical and numerical data in the specified order
    combined_data = np.concatenate((
        preprocessed_text_input[:5],  # Gender, Married, Dependents, Education, Self_Employed
        np.array([applicant_income, coapplicant_income, loan_amount, loan_amount_term]),  # ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
        preprocessed_text_input[5:]   # Property_Area
    ))

    # Reshape data for the model
    input_data = combined_data.reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction result
    if prediction == 1:
        st.success('Predicted Loan Status: Yes')
    else:
        st.error('Predicted Loan Status: No')

