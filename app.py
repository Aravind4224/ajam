
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('house_price_model.pkl')

# Set title
st.title("House Price Prediction App")

# Sidebar inputs
st.sidebar.header("Input Features")

bedrooms = st.sidebar.slider('Bedrooms', 1, 10, 3)
bathrooms = st.sidebar.slider('Bathrooms', 1, 10, 2)
sqft_living = st.sidebar.slider('Sqft Living', 500, 10000, 2000)
floors = st.sidebar.selectbox('Floors', [1, 2, 3])
waterfront = st.sidebar.selectbox('Waterfront (Yes=1, No=0)', [0, 1])
view = st.sidebar.slider('View Rating (0-4)', 0, 4, 0)
grade = st.sidebar.slider('Grade (1-13)', 1, 13, 7)
yr_built = st.sidebar.slider('Year Built', 1900, 2022, 2000)

# Make prediction
input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, floors, waterfront, view, grade, yr_built]],
                          columns=['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'grade', 'yr_built'])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")
