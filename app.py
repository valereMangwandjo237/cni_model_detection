import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uuid
import io
import pytesseract
import shutil
import re

# Configuration de la page
st.set_page_config(
    page_title="Classification CNI/Passeport",
    page_icon="🆔",
    layout="centered"
)

# Titre et description
st.title("🆔 Classification de documents d'identité")
st.markdown("""
Cette application détecte si une image est une **CNI**, un **Passeport**, un **recépissé** ou **autre chose**.
""")

# Sidebar pour les infos supplémentaires
with st.sidebar:
    st.header("À propos")
    st.markdown("""
    Projet IA développé avec:
    - Python
    - Streamlit
    - [Tensorflow§/MobileNet ML]
    """)
    st.write("Auteur: MABOM MANGWANDJO Valère")

# Fonction pour la prédiction (à adapter avec ton vrai modèle)
def detection_visage(img):
  response = 0
  # Charger le modèle pré-entraîné pour détecter les visages
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  
  # Charger l'image
  image = cv2.imread(img)
  
  if image is None:
      raise ValueError(f"Impossible de lire l'image à : {img}")
  
  # Convertir en niveaux de gris
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Détecter les visages
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
  
  # Vérifier si un visage a été trouvé
  if len(faces) > 0:
    response = 1
      
  return response

@st.cache_resource
def load_model_cached():
    return load_model("cni_model_mobilenet_best.keras")

def predict(img_path):
    class_names =  ['new_cni', 'old_cni', 'others', 'passport', 'recepisse']
    predicted_label = ""
    img = image.load_img(img_path, target_size=(224, 224))
    
    model = load_model_cached()
    # Charger et préparer l'image
    img_array = image.img_to_array(img) / 255.0
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Prédiction
    predictions = model.predict(img_array_expanded)
    predicted_index = np.argmax(predictions[0])

    if predicted_index==0 or predicted_index==1:
        predicted_label = "CNI"
    else:
        predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index]
    
    return predicted_label, confidence

def extraction_ocr(img_path, lang="fra+eng"):

    texte = pytesseract.image_to_string(img_path, lang=lang)
    
    nom = re.search(r"([NM][O0][MN]\s*?/\s*SURNAME)[\s:]*\n[\s]*(.+?)(?:\n\n|$)", texte, re.IGNORECASE)
    prenom = re.search(r"(pr[ÉE][nm][0O][MN]S\s*(/|l'|l’)?\s*G[IL]V[FE]N\s*[NM]A[NM][EF]S?)[\s:]*\n[\s]*(.+?)(?:\n\n|$)", texte, re.IGNORECASE)
    date_naiss = re.search(r"DAT[EF]\s*D[EF]\s*[NM]AISSA[NM]C[EF]/DAT[ÉE]\s*OF\s*BIRTH\s*\n\s*(.+)", texte, re.IGNORECASE)
    lieu_naiss = re.search(r"LI[EF]U\s*D[EF]\s*[NM]AISSA[NM]C[EF]/DAT[ÉE]\s*OF\s*BIRTH\s*\n\s*(.+)", texte, re.IGNORECASE)
    
    return {
        'nom': nom.group(1) if nom else 'Non trouvé',
        'prenom': prenom.group(2) if prenom else 'Non trouvé',
        'texte_brut': texte
    }

# Zone de téléchargement
uploaded_file = st.file_uploader(
    "Téléchargez une image de document",
    type=["jpg", "jpeg", "png"],
    help="Format JPG, JPEG ou PNG"
)

col1, col2 = st.columns(2)

with col1:
    if uploaded_file:
        st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
        
with col2:
    if uploaded_file is not None:
        with st.spinner("Analyse en cours..."):
            try:
                # Sauvegarder l'image uploadée temporairement avec un nom de fichier unique
                temp_path = f"temp_upload_{uuid.uuid4().hex}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                if not os.path.exists(temp_path):
                    st.error("Erreur : Fichier temporaire non créé")
                    st.stop() 
                    
                predicted_class, probs = predict(temp_path)
                
                # Affichage des résultats
                st.success("Analyse terminée !")
                st.metric("Confiance", f"{probs*100:.1f}%")
                st.write(temp_path)
                
                # Résultat avec mise en forme conditionnelle
                if predicted_class == "CNI" or predicted_class == "recepisse" or predicted_class == "passport":
                    st.markdown(f"<h2 style='color: #1abc9c;'>📋 {predicted_class}</h2>", unsafe_allow_html=True)
                    text_cni = extraction_ocr(temp_path)
                    with st.expander("📝 Texte extrait de la CNI"):
                        st.write(f"**Nom:** {text_cni['nom']}")
                        st.write(f"**Prenom:** {text_cni['prenom']}")
                        st.write(f"**Texte brut OCR:** {text_cni['texte_brut']}")
                else:
                    st.markdown(f"<h2 style='color: #f39c12;'>❌ {predicted_class}</h2>", unsafe_allow_html=True)
                     
                    
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {str(e)}")
            finally:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            
            
