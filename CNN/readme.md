# Fashion Item Classification using CNN

## Overview

This project is an end-to-end Deep Learning application that classifies fashion items using a Convolutional Neural Network (CNN). The model is trained on the Fashion MNIST dataset and deployed using Streamlit for an interactive user interface.

The application allows users to upload an image and predicts the type of clothing item along with the confidence score.

---

## Dataset

**Dataset:** Fashion MNIST

The Fashion MNIST dataset consists of 70,000 grayscale images of size **28 × 28** belonging to 10 fashion categories.

* Training Images: 60,000
* Test Images: 10,000
* Number of Classes: 10

### Classes

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

---

## CNN Architecture

```text
Input (28 × 28 × 1)

↓

Conv2D (32 filters, 3×3, ReLU)

↓

MaxPooling2D (2×2)

↓

Conv2D (64 filters, 3×3, ReLU)

↓

MaxPooling2D (2×2)

↓

Flatten

↓

Dense (128, ReLU)

↓

Dropout (0.5)

↓

Dense (10, Softmax)
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Streamlit
* Pillow

---

## Project Structure

```text
CNN/

├── cnn.ipynb

├── app/
│   └── app.py

├── models/
│   └── fashion_mnist_cnn.keras

├── test_images/
│   ├── T-shirt_top.png
│   ├── Trouser.png
│   ├── Pullover.png
│   ├── Dress.png
│   ├── Coat.png
│   ├── Sandal.png
│   ├── Shirt.png
│   ├── Sneaker.png
│   ├── Bag.png
│   └── Ankle_boot.png

├── requirements.txt

└── README.md
```

---

## Model Training

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Batch Size: 32
* Epochs: 10
* Activation Functions:

  * ReLU (Hidden Layers)
  * Softmax (Output Layer)

---

## Streamlit Application

The Streamlit application provides:

* Image upload functionality
* Image preprocessing
* Fashion item prediction
* Confidence score display
* Prediction probability visualization

---

## Run Locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run app/app.py
```

---

## Sample Output

```text
Upload Image

↓

Prediction : T-shirt/top

Confidence : 98.42%
```

---
