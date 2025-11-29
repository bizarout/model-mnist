
import streamlit as st 
import tensorflow as tf 


st.set_page_config(page_title="MNIST Prediction App", layout="centered")

st.title("Application de prédiction MNIST")

st.caption("Cette application permet de prédire les chiffres manuscrits en utilisant un modèle pré-entraîné sur le dataset MNIST.")

# chargement du modèle pré-entraîné 

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("./model/mnist_model.h5")
    return model 

with st.spinner("Chargement du modèle..."): 
    model = load_model()
    st.success("Modèle chargé avec succès!") 

from PIL import Image  # conda install anaconda::pillow
import numpy as np 

st.header("📥 Charger une image")

uploaded = st.file_uploader("Choisissez une image 28×28 (ou plus grande)", type=["png", "jpg", "jpeg"])


def preprocess_image(img):
    
    img = img.convert("L").resize((28, 28))
    
    arr = np.array(img, dtype="float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1) 
    return arr, img 

if uploaded: 
    img = Image.open(uploaded)
    st.success("Image téléversée avec succès!")
    
    if st.button("Prétraiter l'image"):
        arr, processed_img = preprocess_image(img)
        st.image(processed_img, caption="Image prétraitée (28x28 en niveaux de gris)", width=150)
        
        

from streamlit_drawable_canvas import st_canvas

canvas = st_canvas(
    fill_color="#000000",
    stroke_width=st.slider("Épaisseur du trait", 5, 40, 18),
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280, width=280,
    drawing_mode="freedraw",
    key="canvas"
)

def preprocess_canvas(canvas_obj):
    if canvas_obj.image_data is not None:
    
        img = Image.fromarray(canvas_obj.image_data.astype("uint8"),mode="RGBA").convert("L").resize((28, 28))
        arr = np.array(img, dtype="float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        return arr, img
    
    

st.subheader("✍️ Choisir la source d'entrée")
source = st.radio("choisissez la source d'entrée:", ["Image chargé", "Dessiner sur le canvas"])

def get_input():

    if source == "Image chargé" and uploaded is not None:
        return preprocess_image(Image.open(uploaded))  
    if source == "Dessiner sur le canvas" and canvas:
        return preprocess_canvas(canvas) if canvas else (None, None)  
    return None, None
    

import matplotlib.pyplot as plt 

def plot_resultats(arr):
    fig ,(ax1) = plt.subplots(1, 1, figsize=(12,8))
    
    bars = ax1.bar(range(10), arr[0], color="skyblue")
    
    chiffre_pred = np.argmax(arr)
    bars[chiffre_pred].set_color("red")
    ax1.set_xticks(range(10))
    ax1.set_xlabel("Chiffres 0-9")
    ax1.set_ylabel("Probabilités")
    ax1.set_title(" Distribution Probabilités de prédiction")
    ax1.grid(True) 
    
    # ajout des valeurs sur les barres 
    for ax in [ax1]:
        for bar in ax.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha="center", va="bottom", fontsize=10)
    
    plt.tight_layout()
    return fig
    
    

# ********************************
if st.button("📈 Lancer la prédiction"):
    arr, processed_img = get_input()          
    if arr is  None:
        st.error("Veuillez fournir une image ou un dessin.")
        
    with st.spinner("Prédiction en cours..."):
        y = model.predict(arr)

        chiffre_pred = int(np.argmax(y))
        conf = float(np.max(y))*100
        st.success(f"Le chiffre  prédit est : {chiffre_pred} avec une confiance de: {conf:.2f})%")
        st.metric("Chiffre prédit", chiffre_pred)
        st.metric("Confiance", f"{conf:.2f} %")
        
        fig = plot_resultats(y)
        st.pyplot(fig)
        
        
        






