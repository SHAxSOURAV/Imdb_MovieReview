#importing the library
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#Mapping of word index back to words(for understanding)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

# Helper function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
#function to preprocess the user input text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#Prediction function
def predict_sentiment(review):
    processed_text = preprocess_text(review)

    prediction =model.predict(processed_text)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

#user input and prediction
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
user_review = st.text_area("Enter your movie review here:")
if st.button('Classify'):

    processed_input = preprocess_text(user_review)

    #Make prediction
    prediction = model.predict(processed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    #display the result
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Score: {prediction[0][0]:.2f}")
else:
    st.write("Please enter a review and click 'Classify' to see the sentiment analysis.")