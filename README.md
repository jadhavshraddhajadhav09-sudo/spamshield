# SpamShield 🛡️

A full-stack web app that uses machine learning to detect whether a text message is **spam** or **ham** (not spam).

## Tech Stack

| Layer    | Technology                                      |
| -------- | ----------------------------------------------- |
| Backend  | Python 3, Flask, flask-cors                     |
| ML       | scikit-learn (TF-IDF + Logistic Regression)     |
| Data     | pandas, SMS Spam Collection dataset             |
| Frontend | Single HTML page, vanilla CSS & JavaScript      |

## Project Structure

```
spamshield/
├── app.py               # Flask server and API routes
├── train_model.py        # Trains and saves the ML model
├── templates/
│   └── index.html        # Home page (frontend)
├── data/
│   └── spam.csv          # SMS Spam Collection dataset
├── model/
│   └── spam_model.joblib # created by train_model.py
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
cd spamshield
pip install -r requirements.txt
```

### 2. Download the dataset

Download the **SMS Spam Collection Dataset** from Kaggle:

> <https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset>

Save the file as `data/spam.csv` inside the `spamshield/` directory.

### 3. Train the model

```bash
python train_model.py
```

This will:
- Load and clean the dataset
- Train a TF-IDF + Logistic Regression pipeline
- Print evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- Save the model to `model/spam_model.joblib`

### 4. Run the app

```bash
python app.py
```

Open your browser at **http://localhost:5000**.

## API Reference

### `POST /api/predict`

Classify a single message.

**Request:**
```json
{ "message": "Congratulations! You've won a $1000 gift card." }
```

**Response:**
```json
{
  "label": "spam",
  "is_spam": true,
  "confidence": 97.42,
  "suspicious_words": ["congratulations", "won", "gift", "card", "claim"]
}
```

**curl example:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "You won a free prize! Click here now."}'
```

### `POST /api/predict-batch`

Classify up to 50 messages at once.

**Request:**
```json
{ "messages": ["Win a free iPhone!", "See you at 3pm."] }
```

**Response:** An array of result objects (same structure as `/api/predict`).

### `GET /api/health`

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /api/history`

Returns the last 20 checks kept in memory:
```json
[
  {
    "preview": "Congratulations! You've won a $1000 gift card. Click http://claim-prize.exampl",
    "label": "spam",
    "confidence": 97.42,
    "timestamp": "2026-10-26T14:00:00+00:00"
  }
]
```

## License

This project is for educational purposes.
