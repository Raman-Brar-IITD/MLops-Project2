import os
import pickle
import mlflow
import nltk

# --- Configuration ---
ASSETS_DIR = "assets"
NLTK_DATA_DIR = os.path.join(ASSETS_DIR, "nltk_data")
MODEL_NAME = "my_model"

def download_nltk_data():
    """
    Downloads all necessary NLTK data packages to a specified directory.
    """
    print(f"Downloading NLTK data to {NLTK_DATA_DIR}...")
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    
    packages = ["punkt", "averaged_perceptron_tagger", "stopwords", "wordnet", "omw-1.4"]
    for package in packages:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            print(f"Downloading package: {package}")
            nltk.download(package, download_dir=NLTK_DATA_DIR)
    
    print("NLTK data download complete.")

def setup_mlflow():
    """
    Configures MLflow tracking URI and credentials from environment variables.
    """
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("FATAL: CAPSTONE_TEST environment variable is not set.")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "Raman-Brar-IITD" # Your DagsHub username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri("https://dagshub.com/Raman-Brar-IITD/MLops-Project2.mlflow")
    print("MLflow configured successfully.")

def download_model_and_vectorizer():
    """
    Downloads the latest model and its vectorizer from MLflow and saves them
    to the assets directory.
    """
    print("Connecting to MLflow to download model and vectorizer...")
    client = mlflow.tracking.MlflowClient()
    
    # Get the latest version, prioritizing the "Production" stage
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not latest_versions:
        print("No 'Production' model found. Falling back to 'None' stage.")
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    
    if not latest_versions:
        raise RuntimeError(f"No models found for '{MODEL_NAME}' in 'Production' or 'None' stages.")
        
    latest_model = latest_versions[0]
    run_id = latest_model.run_id
    print(f"Found model '{MODEL_NAME}' version {latest_model.version} from run_id {run_id}")

    # Create local directory for model and vectorizer
    model_dir = os.path.join(ASSETS_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Download vectorizer
    print("Downloading vectorizer...")
    vectorizer_local_path = client.download_artifacts(run_id, "vectorizer", model_dir)
    print(f"Vectorizer downloaded to {vectorizer_local_path}")

    # Download model
    print("Downloading model...")
    model_uri = f"models:/{MODEL_NAME}/{latest_model.version}"
    mlflow.sklearn.save_model(mlflow.sklearn.load_model(model_uri), path=os.path.join(model_dir, "sklearn_model"))
    print(f"Model downloaded to {os.path.join(model_dir, 'sklearn_model')}")

    print("Model and vectorizer download complete.")

if __name__ == "__main__":
    setup_mlflow()
    download_nltk_data()
    download_model_and_vectorizer()
    print("\nAll assets have been downloaded successfully!")

