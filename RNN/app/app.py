import streamlit as st

import numpy as np

import pickle

import os


from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences



BASE_DIR = os.path.dirname(

    os.path.abspath(__file__)

)


MODELS_DIR = os.path.join(

    BASE_DIR,

    "..",

    "models"

)



model = load_model(

    os.path.join(

        MODELS_DIR,

        "ag_news_simplernn.keras"

    )

)



with open(

    os.path.join(

        MODELS_DIR,

        "tokenizer.pkl"

    ),

    "rb"

) as f:

    tokenizer = pickle.load(

        f

    )



with open(

    os.path.join(

        MODELS_DIR,

        "max_length.pkl"

    ),

    "rb"

) as f:

    max_length = pickle.load(

        f

    )




label_map = {

    0 : "World",

    1 : "Sports",

    2 : "Business",

    3 : "Sci/Tech"

}




st.set_page_config(

    page_title="AG News Classification",

    layout="centered"

)



st.title(

    "News Headline Classification using SimpleRNN"

)



headline = st.text_area(

    "Enter News Headline",

    placeholder="Example: Microsoft launches new software product"

)




if st.button(

    "Predict"

):


    if headline.strip() == "":

        st.warning(

            "Please enter a headline."

        )


    else:


        seq = tokenizer.texts_to_sequences(

            [

                headline

            ]

        )


        pad = pad_sequences(

            seq,

            maxlen=max_length,

            padding="post",

            truncating="post"

        )


        prediction = model.predict(

            pad,

            verbose=0

        )


        pred_class = np.argmax(

            prediction

        )


        confidence = np.max(

            prediction

        )



        st.success(

            f"Prediction : {label_map[pred_class]}"

        )


        st.write(

            f"Confidence : {confidence*100:.2f}%"

        )


        probabilities = prediction[0]


        import pandas as pd



        result_df = pd.DataFrame({

            "Category": [

                "World",

                "Sports",

                "Business",

                "Sci/Tech"

            ],

            "Probability (%)": probabilities * 100

        })

        st.subheader(

            "Prediction Probabilities"

        )

        st.bar_chart(

            result_df.set_index(

                "Category"

            )

        )



        st.markdown(

            "---"

        )

