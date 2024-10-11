# Import necessary libraries for Streamlit
import streamlit as st
import pickle
import re

# Load pre-trained model
model = pickle.load(open('news_classifier.pkl', 'rb'))

# List of basic stopwords (same as in training)
stopwords = set(["the", "and", "a", "is", "in", "it", "for", "to", "of", "that", "on", "this", "with", "as", "by", "at", "from", "an", "be"])

# Preprocess text (same as in training)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters (punctuation, etc.)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

# Streamlit Web App
st.title('Fake News vs. Real News Analyzer')

# Example section for users to understand the app
st.write("""
This web app allows users to analyze whether a given news article is fake or real.
Enter the title and text of a news article, and the app will predict whether it's true or fake.
""")

# Input fields for news article title and text
title = st.text_input('News Title')
text = st.text_area('News Text')

if st.button('Analyze'):
    if title and text:
        # Preprocess and vectorize the text
        full_text = preprocess_text(title + ' ' + text)
        prediction = model.predict([full_text])

        # Predict and display result
        if prediction[0] == 1:
            st.success('This news article is likely True!')
        else:
            st.error('This news article is likely Fake!')
    else:
        st.warning('Please provide both title and text!')

# Footer with example input
st.write('Example Input: Enter a news title and text to get a prediction!')
