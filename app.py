
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake News Detection")
user_input = st.text_area("Enter news content:")

if st.button("Predict"):
    if user_input.strip():
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        label = "Fake" if prediction == 1 else "Real"
        st.success(f"This news is predicted to be: {label}")
    else:
        st.warning("Please enter some text to predict.")
