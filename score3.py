
import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
st.header("Garbage Classification App")

def main():
    st.write("Upload an image to classify.")
    file_uploaded = st.file_uploader("", type=['png', 'jpg', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        result = predict_class(image)
        st.success(f'The image uploaded is: {result}')

def predict_class(image):
    classifier_model = tf.keras.models.load_model(r"/content/drive/MyDrive/Colab Notebooks/garbage_classification_model.hdf5", custom_objects={'KerasLayer': hub.KerasLayer})
    class_names = ['plastic',
                   'battery', 
                   'metal', 
                   'clothes', 
                   'brown-glass', 
                   'shoes', 
                   'biological', 
                   'cardboard', 
                   'green-glass', 
                   'paper', 
                   'trash', 
                   'white-glass']
    test_image = image.resize((224, 224))
    test_image = np.array(test_image)/255.0
    test_image = np.expand_dims(test_image, axis=0)
    predictions = classifier_model.predict(test_image)
    prediction = np.argmax(predictions, axis=1)[0]
    image_class = class_names[prediction]
    return image_class

if __name__ == '__main__':
    main()
