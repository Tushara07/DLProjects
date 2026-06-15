# AG News Classification using SimpleRNN

This project implements a **Simple Recurrent Neural Network (SimpleRNN)** for classifying news headlines into four categories using the AG News dataset.

## Dataset

- Dataset: AG News
- Training Samples: 120,000
- Testing Samples: 7,600
- Classes:
  - World
  - Sports
  - Business
  - Sci/Tech

The dataset is balanced with 30,000 samples in each class.

---

## Model Architecture

The model consists of:

- Embedding Layer
- SimpleRNN Layer
- Dropout Layer
- Dense Layer (ReLU)
- Output Layer (Softmax)

Architecture:

```text
Input Text
     ↓
Tokenizer
     ↓
Padding
     ↓
Embedding
     ↓
SimpleRNN
     ↓
Dropout
     ↓
Dense (ReLU)
     ↓
Dense (Softmax)
     ↓
Predicted Category
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Streamlit

---

## Project Structure

```text
RNN/

├── app/
│   └── app.py

├── models/
│   ├── ag_news_simplernn.keras
│   ├── tokenizer.pkl
│   └── max_length.pkl

├── rnn.ipynb

├── requirements.txt

└── README.md
```

---

## Model Performance

- Training Accuracy: ~88-90%
- Test Accuracy: ~85%

---

## Streamlit Application

The Streamlit application allows users to:

- Enter a news headline
- Predict the news category
- View prediction confidence
- Visualize class probabilities

Example:

```text
Input:
Microsoft launches new software product

Prediction:
Sci/Tech

Confidence:
91.42%
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Tushara07/DLProjects.git
```

Navigate to the project:

```bash
cd DLProjects/RNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/app.py
```

---

## Future Improvements

- Implement LSTM and GRU models
- Compare performance with SimpleRNN
- Add model comparison visualizations
- Deploy using Streamlit Cloud
