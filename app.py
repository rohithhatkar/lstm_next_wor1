
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the LSTM model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('lstm_model.h5')
    return model

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load index_to_word dictionary
@st.cache_resource
def load_index_to_word():
    with open('index_to_word.pkl', 'rb') as f:
        index_to_word = pickle.load(f)
    return index_to_word

# Get max_len from the loaded model's input shape
# Assuming input_length is the second dimension of the input shape
@st.cache_resource
def get_max_len(model):
    # The Embedding layer's input_length is at index 1 of its input_shape
    # The input_shape of the first layer (Embedding) should be (None, max_len)
    max_len = model.layers[0].input_shape[1]
    return max_len

model = load_model()
tokenizer = load_tokenizer()
index_to_word = load_index_to_word()
max_len = get_max_len(model)

def predictor(model, tokenizer, text, max_len):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    padded_sequence = pad_sequences([seq], maxlen=max_len, padding='pre')

    pred = model.predict(padded_sequence, verbose=0)
    pred_index = np.argmax(pred)

    # Handle cases where pred_index might not be in index_to_word
    # This can happen if the model predicts an index outside the known vocabulary
    if pred_index in index_to_word:
        predicted_word = index_to_word[pred_index]
    else:
        predicted_word = None # Or a special token like '[UNK]'

    return predicted_word

def generate_text(model, tokenizer, seed_text, max_len, num_words):
    generated_sequence = seed_text.lower()
    for _ in range(num_words):
        next_word = predictor(model, tokenizer, generated_sequence, max_len)
        if next_word is None:
            break
        generated_sequence += " " + next_word

    return generated_sequence


# Streamlit App Interface
st.title("Text Generation with LSTM")
st.write("Enter a seed text to generate subsequent words.")

seed_text = st.text_input("Seed Text:", "The meaning of life is")
num_words = st.slider("Number of words to generate:", 1, 50, 10)

if st.button("Generate Text"):
    if seed_text:
        with st.spinner("Generating text..."):
            generated_text = generate_text(model, tokenizer, seed_text, max_len, num_words)
            st.success("Text Generated!")
            st.write("**Generated Text:**")
            st.write(generated_text)
    else:
        st.warning("Please enter a seed text to generate!")
