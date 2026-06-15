import streamlit as st
import numpy as np
import os

from PIL import Image

from tensorflow.keras.models import load_model


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)


from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


model = Sequential([

    Conv2D(

        32,

        (3,3),

        activation="relu",

        input_shape=(28,28,1)

    ),

    MaxPooling2D(

        (2,2)

    ),

    Conv2D(

        64,

        (3,3),

        activation="relu"

    ),

    MaxPooling2D(

        (2,2)

    ),

    Flatten(),

    Dense(

        128,

        activation="relu"

    ),

    Dropout(

        0.5

    ),

    Dense(

        10,

        activation="softmax"

    )

])


model.load_weights(

    os.path.join(

        MODELS_DIR,

        "fashion_mnist.weights.h5"

    )

)

class_names = [

    "T-shirt/top",

    "Trouser",

    "Pullover",

    "Dress",

    "Coat",

    "Sandal",

    "Shirt",

    "Sneaker",

    "Bag",

    "Ankle boot"

]


st.set_page_config(

    page_title="Fashion MNIST CNN",

    layout="centered"

)


st.title(
    "Fashion Item Classification using CNN"
)


uploaded_file = st.file_uploader(

    "Upload an Image",

    type=["png", "jpg", "jpeg"]

)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    image = image.convert("L")

    image = image.resize((28, 28))

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = img_array.reshape(
        1,
        28,
        28,
        1
    )

    prediction = model.predict(img_array)

    pred_class = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(
        f"Prediction : {class_names[pred_class]}"
    )

    st.write(
        f"Confidence : {confidence*100:.2f}%"
    )


import pandas as pd

probabilities = prediction[0]

result_df = pd.DataFrame({

    "Class": class_names,

    "Probability (%)": probabilities*100

})

st.subheader("Prediction Probabilities")

st.bar_chart(

    result_df.set_index("Class")

)