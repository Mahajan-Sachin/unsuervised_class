# AnomalyGuard — Credit Card Fraud Detection

Unsupervised Machine Learning project detecting anomalous credit card transactions using an ensemble of **Isolation Forest**, **Local Outlier Factor (LOF)**, and **Autoencoder (TensorFlow/Keras)**.

---

## 🚀 Quick Start

### 1. Download Dataset
Download `creditcard.csv` from Kaggle and place it in `data/`:
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> TensorFlow must be installed in your conda environment separately.

### 3. Generate test CSV files
```bash
python scripts/generate_test_data.py
```
Creates `test_normal.csv`, `test_fraud.csv`, `test_mixed.csv` in `data/`

### 4. Train the models
```bash
python src/train.py
```
Saves trained models to `models/` (~10 min first time)

### 5. Run the web app
```bash
python app.py
```
Open `http://localhost:5000` → Upload a CSV → See results!

---

## 📁 Project Structure
```
├── src/
│   ├── model.py           # Autoencoder architecture (TensorFlow/Keras)
│   ├── train.py           # Train all 3 models & save
│   └── predict.py         # Load models & run predictions
├── scripts/
│   └── generate_test_data.py  # Creates test CSV files from dataset
├── templates/
│   └── index.html         # Web UI (drag & drop CSV upload)
├── static/
│   ├── style.css
│   └── script.js
├── tests/
│   └── test_app.py        # Unit tests
├── data/                  # creditcard.csv + test CSV files
├── models/                # Saved trained models
├── app.py                 # Flask application
└── requirements.txt
```

---

## 🤖 How It Works

| Model | Algorithm | Type |
|---|---|---|
| **Isolation Forest** | Isolates outliers via random trees | Unsupervised (sklearn) |
| **LOF** | Local density-based anomaly detection | Unsupervised (sklearn) |
| **Autoencoder** | High reconstruction error = fraud | Unsupervised Deep Learning (TensorFlow) |

**Decision rule:** If ANY model flags a transaction → marked as Fraud

---

## 🛠 Tech Stack
- Python 3.10, TensorFlow/Keras, scikit-learn, pandas, Flask
