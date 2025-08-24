# Sentiment Analysis MLOps Project

A complete MLOps pipeline for training, deploying, and monitoring a sentiment analysis model. This project is designed to showcase a practical, real-world example of MLOps best practices, making it an ideal portfolio piece for attracting recruiters.

-----
## Data Pipeline
<img width="1710" height="611" alt="image" src="https://github.com/user-attachments/assets/34cee06b-b21a-490a-a97e-ef6cd034b408" />

-----
## Overview 📖

This project demonstrates a full-cycle MLOps workflow for a sentiment analysis task. It includes everything from data ingestion and preprocessing to model training, evaluation, and deployment as a web application. The entire process is automated using **DVC for data versioning** and a **CI/CD pipeline with GitHub Actions** for continuous integration and deployment to **AWS ECR and Hugging Face Spaces**.

What sets this project apart is its focus on automation, reproducibility, and explainability. It's not just about building a model; it's about building a robust, maintainable, and deployable machine learning system.

-----

## Key Features for Recruiters ✨

  * **End-to-End MLOps Pipeline**: This project covers the entire machine learning lifecycle, from data to deployment. This demonstrates a holistic understanding of MLOps principles.
  * **Data and Model Versioning**: I use **DVC** to track datasets and models, ensuring that experiments are reproducible and that you can always roll back to a previous version. This is a critical skill for any serious MLOps role.
  * **Automated CI/CD with GitHub Actions**: The CI/CD pipeline automates the building, testing, and deployment of the application. This shows that I can build automated workflows that save time and reduce errors.
  * **Dual Deployment Strategy**: The application is containerized with **Docker** and deployed to both **AWS ECR** and **Hugging Face Spaces**. This highlights my experience with cloud platforms and containerization technologies.
  * **Interactive Web Application**: I've built a user-friendly web interface with **Flask** that allows for real-time sentiment prediction. This demonstrates my ability to create user-facing applications for machine learning models.
  * **Model Explainability with LIME**: I've integrated **LIME (Local Interpretable Model-agnostic Explanations)** to explain model predictions. This shows that I'm not just building black-box models but that I also care about interpretability and helping users understand why the model makes the decisions it does.

-----

## MLflow 

**MLflow** is central to the MLOps workflow in this project, enabling robust experiment tracking and a streamlined path to production.

  * **📊 Experiment Tracking**: During the model evaluation stage, all relevant information is logged to an MLflow Tracking server hosted on DagsHub. This includes:

      * **Metrics**: Accuracy, Precision, Recall, and AUC are logged to compare model performance across different runs.
      * **Parameters**: The hyperparameters of the model are logged to ensure reproducibility.
      * **Artifacts**: The trained `model.pkl` and the `vectorizer.pkl` are saved as artifacts, linking them directly to the experiment that produced them.

  * **🗂️ Model Registry**: After a model is trained and evaluated, it is registered in the MLflow Model Registry.

      * **Versioning**: The registry versions the model and transitions it to the "Staging" phase.
      * **Promotion**: A separate script (`scripts/promote_model.py`) handles the promotion of a model from "Staging" to "Production" after validation, ensuring only reliable models are deployed.

  * **🚀 Model Deployment**: The Flask web application is designed to be production-aware.

      * The `flask_app/downloader.py` script automatically fetches the latest model version from the "Production" stage in the MLflow Model Registry.
      * This decouples the model training pipeline from the application deployment, allowing for models to be updated in production without requiring a new application build.

-----

## Tech Stack 🛠️

  * **Languages**: Python
  * **Libraries**: scikit-learn, pandas, NLTK, MLflow, Flask
  * **Tools**: DVC, Docker, Git, GitHub Actions
  * **Platforms**: AWS (S3, ECR), Hugging Face Spaces, DagsHub

-----

## Project Structure 📂

```
├── .dvc/                             # DVC configuration files
├── .github/workflows/                # GitHub Actions CI/CD pipeline
├── data/
│   ├── interim/                      # Intermediate, preprocessed data
│   ├── processed/                    # Final, feature-engineered data
│   └── raw/                          # Raw, immutable data
├── docs/                             # Project documentation
├── flask_app/                        # Flask web application source code
│   ├── templates/
│   └── app.py
├── models/                           # Trained models and vectorizers
├── notebooks/                        # Jupyter notebooks for exploration
├── reports/                          # Evaluation metrics and experiment info
├── scripts/                          # Automation scripts (e.g., model promotion)
├── src/                              # Source code for the ML pipeline
│   ├── connections/
│   ├── data/
│   ├── features/
│   ├── logger/
│   └── model/
├── tests/                            # Unit and integration tests
├── .gitignore                        # Files to be ignored by Git
├── Dockerfile                        # Docker configuration for the web app
├── dvc.yaml                          # DVC pipeline definition
├── Makefile                          # Commands for project setup and automation
├── params.yaml                       # Parameters for the DVC pipeline
└── requirements.txt                  # Python dependencies
```

-----

## Installation & Setup ⚙️

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Raman-Brar-IITD/sentiment_analysis_mlops.git
    cd sentiment_analysis_mlops
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your credentials:

    ```
    CAPSTONE_TEST=<your_dagshub_token>
    AWS_ACCESS_KEY_ID=<your_aws_access_key>
    AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
    AWS_REGION=<your_aws_region>
    AWS_ACCOUNT_ID=<your_aws_account_id>
    ECR_REPOSITORY=<your_ecr_repo_name>
    HF_TOKEN=<your_hugging_face_token>
    ```

5.  **Reproduce the DVC pipeline:**
    This will download the data, preprocess it, and train the model.

    ```bash
    dvc repro
    ```

-----

## Usage 🚀

To start the Flask web application locally, run:

```bash
python flask_app/app.py
```

Navigate to `http://127.0.0.1:5000` in your web browser to access the application.

-----

## Testing 🧪

The project includes unit and integration tests to ensure the reliability of the model and the Flask application. To run the tests, use the following command:

```bash
pytest
```

-----

## Future Improvements & Roadmap 🗺️

  * **Advanced Model Monitoring**: Implement a monitoring system to detect data drift and model degradation.
  * **Experimentation with Transformer Models**: Improve accuracy by experimenting with state-of-the-art models like BERT.
  * **Scalable API**: Refactor the Flask application into a more scalable and resilient microservice using FastAPI.
  * **Interactive Dashboard**: Build an interactive dashboard to visualize model performance and explanations.

-----

## Contributing 🤝

Contributions are welcome\! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

-----

## License 📜

This project is licensed under the **MIT License**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for more details.

-----

## Contact 📫

  * **Raman Brar**

  * **GitHub**: [https://github.com/Raman-Brar-IITD](https://www.google.com/search?q=https://github.com/Raman-Brar-IITD)
