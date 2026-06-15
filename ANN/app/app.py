import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# -------------------------------
# Paths
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

# -------------------------------
# Load weights
# -------------------------------

weights = joblib.load(
    os.path.join(
        MODELS_DIR,
        "weights.pkl"
    )
)

W1, b1, W2, b2, W3, b3 = weights


# -------------------------------
# Activation functions
# -------------------------------

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -------------------------------
# ANN Prediction
# -------------------------------

def predict_ann(x):

    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3

    output = sigmoid(z3)

    return output


# -------------------------------
# Load scaler
# -------------------------------

with open(
    os.path.join(
        MODELS_DIR,
        "scaler.pkl"
    ),
    "rb"
) as f:

    scaler = pickle.load(f)


# -------------------------------
# Load feature columns
# -------------------------------

with open(
    os.path.join(
        MODELS_DIR,
        "feature_columns.pkl"
    ),
    "rb"
) as f:

    feature_columns = pickle.load(f)


# -------------------------------
# UI
# -------------------------------

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

    input_scaled = input_scaled.astype(
        np.float32
    )

    prediction = predict_ann(
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