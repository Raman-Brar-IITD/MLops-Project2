from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Text preprocessing functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# Setup MLflow for model loading
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token




    # Set up MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Raman-Brar-IITD/MLops-Project2.mlflow")


# Load model and vectorizer
model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        versions = client.get_latest_versions(model_name, stages=["None"])
    return versions[0].version if versions else None

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    cleaned_text = normalize_text(text)
    features = vectorizer.transform([cleaned_text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
    prediction = model.predict(features_df)[0]
    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
