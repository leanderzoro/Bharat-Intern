import streamlit as st
import pickle

# Load the SVM model and TF-IDF vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define function to preprocess input text
def preprocess_text(text):
    # Apply the same preprocessing steps used during training
    ...

# Define function to make predictions
def predict(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    
    # Convert preprocessed text to TF-IDF vector
    tfidf_vector = vectorizer.transform([preprocessed_text])
    
    # Make prediction using the loaded SVM model
    prediction = model.predict(tfidf_vector)
    
    return prediction

# Streamlit app code
st.title('Spam Detection App')
text_input = st.text_input('Enter a message:')

if st.button('Predict'):
    prediction = predict(text_input)
    st.write('Prediction:', prediction)