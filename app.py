import joblib
from PIL import Image
import pandas as pd
import streamlit as st

# loading pre-trained classifier
model = joblib.load('classification_model.sav')

# main App layout
st.header("ML Iris Classificator:")
image = Image.open('irises.jpeg')
st.image(image, use_column_width=True)
st.write("Feed flower properties for automatic classification:")

# Define slider ranges based on Iris dataset feature ranges
SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 8.0, step=0.1)
SepalWidthCm = st.slider('SepalWidthCm:', 2.0, 4.5, step=0.1)
PetalLengthCm = st.slider('PetalLengthCm:', 1.0, 7.0, step=0.1)
PetalWidthCm = st.slider('PetalWidthCm:', 0.1, 2.5, step=0.1)

data = {'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm}

features = pd.DataFrame(data, index=[0])

pred_proba = model.predict_proba(features)

st.subheader('Class probabilities:') 

# Results
st.write('**Setosa: `{}`%**'.format(round(pred_proba[0][0]*100, 2)))
st.write('**Versicolor: `{}`%**'.format(round(pred_proba[0][1]*100, 2)))
st.write('**Virginica: `{}`%**'.format(round(pred_proba[0][2]*100, 2)))
