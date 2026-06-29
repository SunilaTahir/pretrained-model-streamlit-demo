import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Pretrained Model Demo", page_icon="🤖")

st.title("Pretrained Sentiment Classifier")

st.write(
    "This app uses a pretrained Hugging Face model to predict whether a sentence is positive or negative."
)

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

user_text = st.text_area(
    "Enter text:",
    "I really enjoyed this NLP lab because the steps were simple and clear."
)

if st.button("Predict Sentiment"):
    if user_text.strip():
        result = classifier(user_text)[0]
        st.success(f"Label: {result['label']}")
        st.info(f"Confidence: {result['score']:.4f}")
    else:
        st.warning("Please enter some text first.")
