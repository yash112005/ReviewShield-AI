import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack

# Load model and vectorizer
model = joblib.load("review_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="ReviewShield AI", page_icon="🛡️")

st.title("🛡️ ReviewShield AI")
st.write("Detect whether a customer review is Genuine or Fake using Machine Learning.")

review = st.text_area("Customer Review")

rating = st.slider("Rating", 1, 5, 5)

verified = st.selectbox("Verified Purchase", ["Yes", "No"])

total_reviews = st.number_input(
    "Reviewer Total Reviews",
    min_value=0,
    value=1
)

helpful_votes = st.number_input(
    "Helpful Votes",
    min_value=0,
    value=0
)

if st.button("Predict"):

    verified_num = 1 if verified == "Yes" else 0

    text_features = vectorizer.transform([review])

    meta = [[
        rating,
        verified_num,
        total_reviews,
        helpful_votes
    ]]

    final_input = hstack([text_features, meta])

    prediction = model.predict(final_input)[0]

    if prediction == 1:
      st.error("🚨 Fake Review Detected")
    else:
      st.success("✅ Genuine Review")