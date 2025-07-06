import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Class names as per the model training
CLASS_NAMES = ['pituitary', 'notumor', 'meningioma', 'glioma']

# Load the model (cache to avoid reloading on every run)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('brain_tumor_model.h5')

model = load_model()

st.title('Brain Tumor Classification from X-ray')
st.write('Upload a brain X-ray image to predict the type of brain tumor.')

# --- Sidebar: Project Info and Social Links ---
st.sidebar.title('About This Project')
st.sidebar.info(
    'This app uses deep learning to classify brain tumors from X-ray images. Upload an image to get a prediction of the tumor type.\n\n'
    'Model trained on a public brain tumor MRI dataset with four classes: pituitary, no tumor, meningioma, and glioma.'
)
st.sidebar.markdown('---')
st.sidebar.markdown('**Connect with me:**')
st.sidebar.markdown('[LinkedIn](https://www.linkedin.com/in/prititaliya)')
st.sidebar.markdown('[GitHub](https://github.com/prititaliya)')

uploaded_file = st.file_uploader('Choose a brain X-ray image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    pred_probs = preds[0]

    st.subheader('Prediction:')
    st.write(f'**{pred_class.capitalize()}**')

    st.subheader('Prediction Probabilities:')
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f'{class_name.capitalize()}: {pred_probs[i]*100:.2f}%') 