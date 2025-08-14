FROM python:3.10-slim
WORKDIR /app

# Copy all application code, requirements, and pre-downloaded assets
COPY flask_app/ /app/
COPY requirements.txt .
COPY assets/ /app/assets/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Start the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]