import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from tensorflow.keras.models import load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)


model = load_model(
    os.path.join(
        MODELS_DIR,
        "breast_cancer_ann.keras"
    )
)


with open(
    os.path.join(
        MODELS_DIR,
        "scaler.pkl"
    ),
    "rb"
) as f:

    scaler = pickle.load(f)


with open(
    os.path.join(
        MODELS_DIR,
        "feature_columns.pkl"
    ),
    "rb"
) as f:

    feature_columns = pickle.load(f)


st.title("Breast Cancer Prediction using ANN")


radius_mean = st.number_input(
    "Radius Mean",
    value=14.0
)

texture_mean = st.number_input(
    "Texture Mean",
    value=20.0
)

perimeter_mean = st.number_input(
    "Perimeter Mean",
    value=90.0
)

area_mean = st.number_input(
    "Area Mean",
    value=600.0
)

smoothness_mean = st.number_input(
    "Smoothness Mean",
    value=0.1,
    format="%.4f"
)

compactness_mean = st.number_input(
    "Compactness Mean",
    value=0.1,
    format="%.4f"
)

concavity_mean = st.number_input(
    "Concavity Mean",
    value=0.1,
    format="%.4f"
)

concave_points_mean = st.number_input(
    "Concave Points Mean",
    value=0.05,
    format="%.4f"
)


if st.button("Predict"):

    input_data = pd.DataFrame(
        [[
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean
        ]],
        columns=feature_columns
    )

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )

    probability = prediction[0][0]


    if probability > 0.5:

        st.error(
            f"Prediction : Malignant\n\n"
            f"Probability : {probability*100:.2f}%"
        )

    else:

        st.success(
            f"Prediction : Benign\n\n"
            f"Probability : {(1-probability)*100:.2f}%"
        )

    
    st.subheader("Input Feature Values")

    chart_data = pd.DataFrame({
        "Feature": [
            "Radius",
            "Texture",
            "Perimeter",
            "Area",
            "Smoothness",
            "Compactness",
            "Concavity",
            "Concave Points"
        ],
        "Value": [
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean
        ]
    })

    st.bar_chart(
        chart_data.set_index("Feature")
    )
