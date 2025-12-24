import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Page configuration
st.set_page_config(page_title="LSTM Text Generator")

# Load resources
@st.cache_resource
def load_model():
    try:
        with open('lstm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

@st.cache_resource
def load_index_to_word():
    try:
        with open('index_to_word.pkl', 'rb') as f:
            index_to_word = pickle.load(f)
        return index_to_word
    except Exception as e:
        st.error(f"Error loading word index: {e}")
        return None

# Load everything
model = load_model()
tokenizer = load_tokenizer()
index_to_word = load_index_to_word()

# Get max_len
if model:
    max_len = model.layers[0].input_shape[1]
else:
    max_len = 20  # Default fallback

# Text generation functions
def predict_next_word(text):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])
    if not seq[0]:
        return None
    
    padded = pad_sequences(seq, maxlen=max_len, padding='pre')
    pred = model.predict(padded, verbose=0)
    pred_index = np.argmax(pred)
    
    return index_to_word.get(pred_index, None)

def generate_text(seed_text, num_words=10):
    generated = seed_text
    for _ in range(num_words):
        next_word = predict_next_word(generated)
        if next_word is None:
            break
        generated += " " + next_word
    return generated

# Streamlit UI
st.title("📝 LSTM Text Generator")
st.write("Enter a starting phrase to generate text:")

seed_text = st.text_input("Seed text:", "The meaning of life is")
num_words = st.slider("Words to generate:", 1, 50, 10)

if st.button("Generate"):
    if model and tokenizer and index_to_word:
        with st.spinner("Generating..."):
            result = generate_text(seed_text, num_words)
            st.success("Generated Text:")
            st.write(result)
    else:
        st.error("Model not loaded properly!")
