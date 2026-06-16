import streamlit as st

from utils.logistic_predict import predict_logistic
from utils.lstm_predict import predict_lstm
from utils.finbert_predict import predict_finbert

st.title("Financial News Sentiment Analyzer")

text = st.text_area(
    "Enter financial news headline"
)

model_choice = st.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "LSTM",
        "FinBERT"
    ]
)

if st.button("Predict"):
    if not text:
        st.warning("Please enter a headline to analyze.")
    with st.spinner(f"Running {model_choice} model..."):
        if model_choice == "Logistic Regression":
            result = predict_logistic(text)

        elif model_choice == "LSTM":
            result = predict_lstm(text)

        else:
            result = predict_finbert(text)

        st.success(f"Sentiment: {result}")