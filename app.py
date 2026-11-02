"""
app.py — Flask server for SpamShield.

Routes:
    GET  /               → serves the frontend
    POST /api/predict    → single message classification
    POST /api/predict-batch → batch classification (up to 50 messages)
    GET  /api/health     → server, model & Supabase health check
    GET  /api/history    → last 20 prediction results (Supabase or in-memory)
"""

import os
import re
import string
from collections import deque
from datetime import datetime, timezone

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.joblib")

# ---------------------------------------------------------------------------
# Environment variables (.env)
# ---------------------------------------------------------------------------
# Attempt loading .env from current dir, BASE_DIR, and parent workspace dir
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ---------------------------------------------------------------------------
# Supabase client setup
# ---------------------------------------------------------------------------
supabase: Client | None = None

if (
    not SUPABASE_URL
    or not SUPABASE_KEY
    or SUPABASE_URL.strip() == ""
    or SUPABASE_KEY.strip() == ""
    or SUPABASE_URL.startswith("PASTE_")
    or SUPABASE_KEY.startswith("PASTE_")
):
    print(
        "[WARNING] SUPABASE_URL or SUPABASE_KEY is missing or unconfigured.\n"
        "          Falling back to in-memory history."
    )
else:
    try:
        supabase = create_client(SUPABASE_URL.strip(), SUPABASE_KEY.strip())
        print(f"[OK] Supabase client initialized for {SUPABASE_URL.strip()}")
    except Exception as e:
        print(
            f"[WARNING] Failed to initialize Supabase client: {e}.\n"
            "          Falling back to in-memory history."
        )
        supabase = None

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
model = None

if os.path.isfile(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
else:
    print(
        "[WARNING] Model file not found at", MODEL_PATH,
        "\n   Run `python train_model.py` first to train and save the model.",
    )

# ---------------------------------------------------------------------------
# In-memory history fallback (thread-safe enough for a single-process dev server)
# ---------------------------------------------------------------------------
MAX_HISTORY = 20
history: deque = deque(maxlen=MAX_HISTORY)

# ---------------------------------------------------------------------------
# Text cleaning (must match train_model.py)
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
NUMBER_RE = re.compile(r"\b\d+\b")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Normalise a message the same way train_model.py does."""
    text = text.lower()
    text = URL_RE.sub("urltoken", text)
    text = EMAIL_RE.sub("emailtoken", text)
    text = NUMBER_RE.sub("numbertoken", text)
    text = text.translate(PUNCT_TABLE)
    return text.strip()


# ---------------------------------------------------------------------------
# Suspicious-word extraction
# ---------------------------------------------------------------------------
def get_suspicious_words(message: str, top_n: int = 8) -> list[str]:
    """Return up to *top_n* words from *message* that most strongly push
    the prediction toward the 'spam' class, ranked by the model's learned
    logistic regression coefficients.
    """
    if model is None:
        return []

    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    # For binary LogisticRegression, coef_ has shape (1, n_features).
    # Positive coefficients push toward the second class (alphabetically),
    # which is "spam" (since "spam" > "ham").
    coefs = clf.coef_[0]                 # shape: (n_features,)

    # Transform the cleaned message to TF-IDF
    cleaned = clean_text(message)
    vec = tfidf.transform([cleaned])     # sparse (1, n_features)

    # Element-wise contribution of each feature
    contributions = np.asarray(vec.multiply(coefs).todense()).flatten()

    # Map feature indices back to vocabulary
    feature_names = np.array(tfidf.get_feature_names_out())

    # Pick the top-N positive contributors (pushing toward spam)
    positive_mask = contributions > 0
    if not positive_mask.any():
        return []

    indices = np.argsort(contributions)[::-1]
    top_words = []
    for idx in indices:
        if contributions[idx] <= 0:
            break
        word = feature_names[idx]
        # Only include single words (skip bigrams for cleaner output)
        if " " not in word and word not in ("urltoken", "emailtoken", "numbertoken"):
            top_words.append(word)
        if len(top_words) >= top_n:
            break

    return top_words


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_single(message: str) -> dict:
    """Run the model on a single message and return a result dict."""
    cleaned = clean_text(message)
    label: str = model.predict([cleaned])[0]
    probas = model.predict_proba([cleaned])[0]
    spam_idx = list(model.classes_).index("spam")
    confidence = float(probas[spam_idx] * 100) if label == "spam" else float((1 - probas[spam_idx]) * 100)

    suspicious = get_suspicious_words(message) if label == "spam" else []

    return {
        "label": label,
        "is_spam": label == "spam",
        "confidence": round(confidence, 2),
        "suspicious_words": suspicious,
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_message(value) -> tuple[str | None, str | None]:
    """Validate a single message value. Returns (message, error)."""
    if value is None or not isinstance(value, str):
        return None, "message must be a non-empty string."
    text = value.strip()
    if not text:
        return None, "message must be a non-empty string."
    if len(text) > 5000:
        return None, "message must be 5 000 characters or fewer."
    return text, None


# ---------------------------------------------------------------------------
# Supabase storage helpers
# ---------------------------------------------------------------------------
def save_prediction(message_preview: str, label: str, confidence: float, created_at: str) -> None:
    """Insert a single prediction into the Supabase 'predictions' table.
    Fails gracefully without interrupting request flow.
    """
    if supabase is None:
        return
    try:
        supabase.table("predictions").insert({
            "message_preview": message_preview,
            "label": label,
            "confidence": confidence,
            "created_at": created_at,
        }).execute()
    except Exception as e:
        print(f"[WARNING] Failed to save prediction to Supabase: {e}")


def save_predictions_batch(records: list[dict]) -> None:
    """Insert multiple predictions into the Supabase 'predictions' table.
    Fails gracefully without interrupting request flow.
    """
    if supabase is None or not records:
        return
    try:
        supabase.table("predictions").insert(records).execute()
    except Exception as e:
        print(f"[WARNING] Failed to save batch predictions to Supabase: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """Classify a single message as spam or ham.

    Expects JSON: {"message": "..."}
    Returns JSON with label, is_spam, confidence, and suspicious_words.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    msg, err = validate_message(data.get("message"))
    if err:
        return jsonify({"error": err}), 400

    result = predict_single(msg)

    preview = msg[:80]
    now_iso = datetime.now(timezone.utc).isoformat()

    # Always keep in-memory history updated
    history.appendleft({
        "preview": preview,
        "label": result["label"],
        "confidence": result["confidence"],
        "timestamp": now_iso,
    })

    # Record to Supabase (fails gracefully if any error occurs)
    save_prediction(preview, result["label"], result["confidence"], now_iso)

    return jsonify(result)


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    """Classify up to 50 messages at once.

    Expects JSON: {"messages": ["...", ...]}
    Returns a JSON array of result objects.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    messages = data.get("messages")
    if not isinstance(messages, list):
        return jsonify({"error": "'messages' must be an array."}), 400
    if len(messages) > 50:
        return jsonify({"error": "Batch size must be 50 or fewer."}), 400
    if len(messages) == 0:
        return jsonify({"error": "'messages' array must not be empty."}), 400

    results = []
    db_records = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for i, raw in enumerate(messages):
        msg, err = validate_message(raw)
        if err:
            return jsonify({"error": f"messages[{i}]: {err}"}), 400
        result = predict_single(msg)
        results.append(result)

        preview = msg[:80]
        history.appendleft({
            "preview": preview,
            "label": result["label"],
            "confidence": result["confidence"],
            "timestamp": now_iso,
        })
        db_records.append({
            "message_preview": preview,
            "label": result["label"],
            "confidence": result["confidence"],
            "created_at": now_iso,
        })

    # Record batch to Supabase (fails gracefully if any error occurs)
    save_predictions_batch(db_records)

    return jsonify(results)


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "supabase_connected": supabase is not None,
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return the last 20 prediction results. Reads from Supabase if available,
    falling back to in-memory history if Supabase is unconfigured or errors.
    """
    if supabase is not None:
        try:
            response = (
                supabase.table("predictions")
                .select("message_preview, label, confidence, created_at")
                .order("created_at", desc=True)
                .limit(20)
                .execute()
            )
            items = []
            for row in (response.data or []):
                items.append({
                    "preview": row.get("message_preview") or row.get("preview", ""),
                    "label": row.get("label", ""),
                    "confidence": row.get("confidence", 0.0),
                    "timestamp": row.get("created_at") or row.get("timestamp", ""),
                })
            return jsonify(items)
        except Exception as e:
            print(f"[WARNING] Failed to fetch history from Supabase: {e}. Falling back to in-memory history.")

    return jsonify(list(history))


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(_e):
    """Return JSON for 404 errors instead of HTML."""
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def internal_error(_e):
    """Return JSON for 500 errors instead of HTML."""
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
