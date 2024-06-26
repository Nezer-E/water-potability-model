import numpy as np 
import pandas as pd 
#import matplotlib.pyplot as plt
#import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import streamlit as st


def inputprocessor():
#User Input 
    st.sidebar.header("LOAN ELIGIBITY MODEL INPUT PARAMETERS")

    Gender =st.sidebar.selectbox(label= "Select Gender", options= ["Male", "Female"])
    Marital_status = st.sidebar.selectbox(label = "Married", options= ["Yes", "No"])
    Dependents = st.sidebar.slider("Number of Dependents", 0, 7, 3)
    Education = st.sidebar.selectbox(label="Education", options=["Not Graduate", "Graduate"])
    Self_employed = st.sidebar.selectbox(label="Employed?", options= ["Yes", "No"])
    Applicant_Income = st.sidebar.text_input("Applicant's Income", 1000)
    Co_applicant_income = st.sidebar.text_input("Co-applicant's Income", 2000)
    Loan_Amount = st.sidebar.text_input("Loan Amount", 500)
    Loan_Term = st.sidebar.text_input("loan Term", 120)
    credit_histroy = st.sidebar.selectbox(label="Credit History", options=["Good", "Bad"])
    Property_area = st.sidebar.selectbox(label = "Property Area", options = ["Urban", "Rural", "Semiurban"])
    button = st.sidebar.button("Process")

    test_data = pd.DataFrame({
        'Gender': [Gender],
        'Married': [Marital_status],
        'Dependents': [Dependents],
        'Education': [Education],
        'Self_Employed': [Self_employed],
        'ApplicantIncome': [Applicant_Income],
        'CoapplicantIncome': [Co_applicant_income],
        'LoanAmount': [Loan_Amount],
        'Loan_Amount_Term': [Loan_Term],
        'Credit_History': [credit_histroy],
        'Property_Area': [Property_area]
    })
    # main output region
    st.header("LOAN ELIGIBITY MODEL")
    st.write(test_data)
    train_data = pd.read_csv('loantrain.csv')
    train_data = train_data.drop('Loan_ID', axis = 1)

    train_data = train_data.replace('Male', 1)
    train_data = train_data.replace('Female', 0)
    train_data = train_data.replace('Yes', 1)
    train_data = train_data.replace('No', 0)
    train_data = train_data.replace('Graduate', 1)
    train_data = train_data.replace('Not Graduate', 0)
    train_data = train_data.replace('Urban', 2)
    train_data = train_data.replace('Semiurban', 1 )
    train_data = train_data.replace('Rural', 0)
    train_data = train_data.replace('Y', 1)
    train_data = train_data.replace('N', 0)
    train_data = train_data.replace('3+', 3)



    #test_data replacement of input values
    test_data = test_data.replace('Male', 1)
    test_data = test_data.replace('Female', 0)
    test_data = test_data.replace('Married', 1)
    test_data = test_data.replace('Single', 0)
    test_data = test_data.replace('Graduate', 1)
    test_data = test_data.replace('Not Graduate', 0)
    test_data = test_data.replace('Urban', 2)
    test_data = test_data.replace('Semiurban', 1 )
    test_data = test_data.replace('Rural', 0)
    test_data = test_data.replace('3+', 3)
    test_data = test_data.replace('Yes', 1)
    test_data = test_data.replace('NO', 0)
    #credit history modification
    test_data = test_data.replace('Good', 1)
    test_data = test_data.replace('Bad', 0)

    #summary.to_csv('summaryofloaneligibility.csv')

    #handling missing values 

    train_data['Gender'] = train_data['Gender'].fillna(1)
    train_data['Married'] = train_data['Married'].fillna(1)
    train_data['Self_Employed'] = train_data['Self_Employed'].fillna(0) 
    train_data['LoanAmount'] = train_data['LoanAmount'].fillna(146.412162162162)
    train_data['Loan_Amount_Term'] = train_data['Loan_Amount_Term'].fillna(342)
    train_data['Credit_History'] = train_data['Credit_History'].fillna(1)

    summary = train_data.describe()
    st.subheader("Summary of training dataset")
    st.write(summary)
    st.subheader("Results will appear in this section")

    #extracting dependent and independent variable

    X_train = train_data.drop('Loan_Status', axis = 1)
    Y_train = pd.DataFrame({
        'Loan_Status': train_data['Loan_Status']
    })

    #conversion to numpy array
    if button:

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        test_data = np.array(test_data)

        #feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        test_data = scaler.transform(test_data)

        params = {
            'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            'max_depth':[1,2,3,4,5,6,7,8,9,10],
            'n_estimators': [10,20,30,40,50,60,70,80,90,100],
            'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }

        #accuracy is 77 percent and max = 85.4%

        classifier = XGBClassifier(learning_rate = 0.3, n_estimators = 60, max_depth = 1, subsample = 1.0, colsample_bytree = 0.8 )

        classifier.fit(X_train, Y_train)

        predicted_value = classifier.predict(test_data)

        if predicted_value == [1]:
            st.write(f'The predicted value is {predicted_value}. Customer is eligible for loan')
        else: 
            st.write(f'The predicted value is {predicted_value}. Customer is not eligible for loan')

inputprocessor()
