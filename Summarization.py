import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

# Charger le modèle et le tokenizer
@st.cache_resource
def load_model():
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    return model, tokenizer

model, tokenizer = load_model()

# Fonction pour générer du texte avec BART
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Interface utilisateur Streamlit
st.title("Générateur de texte avec BART")
st.write("Entrez un texte ci-dessous pour générer une réponse :")

prompt = st.text_area("Votre texte", "Entrez le texte ici")

if st.button("Générer le texte"):
    if prompt:
        result = generate_text(prompt)
        st.write("Texte généré :")
        st.write(result)
    else:
        st.write("Veuillez entrer un texte pour générer la réponse.")
