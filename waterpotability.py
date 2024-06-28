import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
#import seaborn as sns
import streamlit as st

def inputprocessor():
    data = pd.read_csv("waterpotability.csv")
    st.sidebar.header("WATER POTABILITY INPUT PARAMETERS")

    pH = st.sidebar.slider("pH", 0, 14, 7)
    Hardness = st.sidebar.text_input("Hardness", 129.5)
    Solids = st.sidebar.text_input("Solids", 20791)
    Chloramines = st.sidebar.text_input("Chloramines", 7.3)
    Sulfate = st.sidebar.text_input("Sulfate", 368.7)
    Conductivity = st.sidebar.text_input("Conductivity", 592.8)
    Organic_carbon = st.sidebar.text_input("Organic_Carbon", 13.7)
    Trihalomethanes = st.sidebar.text_input("Trihalomethanes", 56.3)
    Turbidity = st.sidebar.text_input("Turbidity", 4.5)
    button = st.sidebar.button("Process")

    test_data = pd.DataFrame({
        'ph': [pH],	
        'Hardness':	[Hardness],
        'Solids': [Solids],
        'Chloramines': [Chloramines],
        'Sulfate': [Sulfate],
        'Conductivity': [Conductivity],
        'Organic_carbon': [Organic_carbon],
        'Trihalomethanes': [Trihalomethanes],
        'Turbidity': [Turbidity],
    })
    st.header("WATER POTABILITY MODEL")
    st.subheader("User Input Data")
    st.write(test_data)
    st.subheader("Results will appear in this section")

    # filling missing values 
    data['ph'] = data['ph'].fillna(7.080795)
    data['Hardness'] = data['Hardness'].fillna(196.369496)
    data['Solids'] = data['Solids'].fillna(22014.0)
    data['Chloramines'] = data['Chloramines'].fillna(7.122277)
    data['Sulfate'] = data['Sulfate'].fillna(333.7758)
    data['Conductivity'] = data['Conductivity'].fillna(426.2051107)
    data['Organic_carbon'] = data['Organic_carbon'].fillna(14.284970)
    data['Trihalomethanes'] = data['Trihalomethanes'].fillna(66.396293)
    data['Turbidity'] = data['Turbidity'].fillna(3.966786)
    
    #summary = data.describe()
    X_train = data.drop('Potability', axis = 1)
    Y_train = pd.DataFrame({
        'Potability': data['Potability']
    })

    # test_data to be written here
    
    #conversion to numpy array
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    test_data = np.array(test_data)

    #Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    test_data = scaler.transform(test_data)

    params= {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'max_depth': [1,2,3,4,5,6,7,8,9,10],
        'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'n_estimators': [10,20,30,40,50,60,70,80,90,100],
    }

    classifier = XGBClassifier(learning_rate = 0.15, max_depth =6, n_estimators = 10, subsample = 0.4, colsample_bytree = 0.5)
    classifier.fit(X_train, Y_train)
    predicted_results = classifier.predict(test_data)

    if button:
        if predicted_results == [0]:
            st.write('Sample is not Potable')
        elif predicted_results == [1]:
            st.write('Sample is Potable')

inputprocessor()
