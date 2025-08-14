import os
import pickle
import re
import string
import warnings

import flask
import mlflow.sklearn
import nltk
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Initial Setup & Configuration ---

# Suppress warnings for a cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# --- Asset Loading ---

ASSETS_DIR = "assets"
NLTK_DATA_DIR = os.path.join(ASSETS_DIR, "nltk_data")
MODEL_DIR = os.path.join(ASSETS_DIR, "model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer", "vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "sklearn_model")

# Point NLTK to the local data directory
nltk.data.path.append(NLTK_DATA_DIR)

# --- Text Preprocessing Functions ---

# Load NLTK resources from the local path
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))
PUNCTUATION_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_REGEX = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    """
    Applies a series of text normalization steps using pre-loaded resources.
    """
    text = text.lower()
    text = URL_REGEX.sub("", text)
    text = PUNCTUATION_REGEX.sub(" ", text)
    text = "".join([char for char in text if not char.isdigit()])
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])
    text = " ".join([LEMMATIZER.lemmatize(word) for word in text.split()])
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text

# --- LIME Explainer Setup ---

def create_predictor_for_lime(model, vectorizer):
    """
    Creates a prediction function compatible with LIME's TextExplainer.
    """
    def predictor(texts):
        cleaned_texts = [normalize_text(t) for t in texts]
        features = vectorizer.transform(cleaned_texts)
        return model.predict_proba(features)
    return predictor

# --- Flask Application ---

app = flask.Flask(__name__)

# Load model and vectorizer once from local files when the app starts
print("Loading local assets...")
with open(VECTORIZER_PATH, "rb") as f:
    VECTORIZER = pickle.load(f)
print("Vectorizer loaded successfully.")

MODEL = mlflow.sklearn.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Create the LIME explainer
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
EXPLAINER = LimeTextExplainer(
    class_names=["Negative", "Positive"],
    split_expression=lambda x: TOKEN_PATTERN.findall(x)
)
LIME_PREDICTOR = create_predictor_for_lime(MODEL, VECTORIZER)
print("LIME explainer created.")

@app.route("/")
def home():
    """Renders the main page."""
    return flask.render_template("index.html", result=None, explanation=None)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles the prediction request using the pre-loaded model.
    """
    text = flask.request.form.get("text", "").strip()
    
    if not text:
        return flask.render_template("index.html", result="empty", explanation=None)

    # Generate the explanation
    explanation = EXPLAINER.explain_instance(
        text, LIME_PREDICTOR, num_features=10
    )
    
    # Get the final prediction
    cleaned_text = normalize_text(text)
    features = VECTORIZER.transform([cleaned_text])
    prediction_code = MODEL.predict(features)[0]
    prediction_label = "Positive" if prediction_code == 1 else "Negative"

    return flask.render_template(
        "index.html",
        text=text,
        result=prediction_label,
        explanation=explanation.as_html()
    )

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

