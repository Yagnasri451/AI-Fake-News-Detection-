import os
import streamlit as st
import joblib

from preprocess import clean_text


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# --------------------------------
# Title
# --------------------------------

st.title("📰 AI-Based Fake News Detection")

st.write(
    "Enter a news article below and the AI model will "
    "predict whether the news is Fake or Real."
)


# --------------------------------
# Load model
# --------------------------------

MODEL_PATH = "model/fake_news_model.pkl"

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model not found. Please run train_model.py first."
    )

    st.stop()


model = joblib.load(MODEL_PATH)


# --------------------------------
# User input
# --------------------------------

news_text = st.text_area(
    "Enter news article:",
    height=250,
    placeholder="Paste the news article here..."
)


# --------------------------------
# Prediction
# --------------------------------

if st.button("🔍 Detect Fake News"):

    if not news_text.strip():

        st.warning(
            "Please enter some news text."
        )

    else:

        cleaned_news = clean_text(news_text)

        prediction = model.predict(
            [cleaned_news]
        )[0]

        probability = model.predict_proba(
            [cleaned_news]
        )[0]

        confidence = max(probability) * 100


        # 1 = REAL
        # 0 = FAKE

        if prediction == 1:

            st.success(
                "✅ This news is predicted to be REAL."
            )

        else:

            st.error(
                "❌ This news is predicted to be FAKE."
            )


        st.info(
            f"Prediction confidence: {confidence:.2f}%"
        )