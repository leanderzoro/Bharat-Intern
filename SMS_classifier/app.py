import streamlit as st

# Load the SVM model and TF-IDF vectorizer
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Define function to preprocess input text
def preprocess_text(text):
    # Apply the same preprocessing steps used during training
    ...

# Define function to make predictions
def predict(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    
    # Convert preprocessed text to TF-IDF vector
    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])
    
    # Make prediction using the loaded SVM model
    prediction = svm_model.predict(tfidf_vector)
    
    return prediction

# Streamlit app code
st.title('Spam Detection App')
text_input = st.text_input('Enter a message:')

if st.button('Predict'):
    prediction = predict(text_input)
    st.write('Prediction:', prediction)