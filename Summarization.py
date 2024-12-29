import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the pre-trained BART model and tokenizer from Hugging Face
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Function to summarize the input text
def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app layout
st.title("Text Summarization using BART")
st.write("This is a simple text summarization app using the BART model. Enter your text below and get a summary.")

# Text input field for the user
input_text = st.text_area("Enter Text", "Paste or type your text here...")

if st.button("Generate Summary"):
    if input_text.strip():
        summary = summarize_text(input_text)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
