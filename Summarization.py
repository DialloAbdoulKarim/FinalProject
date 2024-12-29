import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Charger le modèle pré-entraîné BART et le tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Fonction de génération de texte
def generate_text(input_text, max_length=100, num_beams=5):
    # Tokeniser le texte d'entrée
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Générer le texte avec des paramètres de contrôle
    output = model.generate(inputs['input_ids'], 
                            max_length=max_length, 
                            num_beams=num_beams, 
                            no_repeat_ngram_size=2, 
                            early_stopping=True)
    
    # Décoder et retourner le texte généré
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Titre de l'application Streamlit
st.title("Générateur de Texte avec BART")

# Entrée utilisateur pour le texte d'entrée
input_text = st.text_area("Entrez un texte de départ", "Once upon a time in a land far away")

# Slider pour la longueur du texte généré
max_length = st.slider("Longueur du texte généré", 50, 200, 100)

# Choisir le nombre de beams pour la recherche
num_beams = st.slider("Nombre de beams pour la génération", 1, 10, 5)

# Bouton pour générer le texte
if st.button("Générer le texte"):
    generated_text = generate_text(input_text, max_length, num_beams)
    st.subheader("Texte généré :")
    st.write(generated_text)
