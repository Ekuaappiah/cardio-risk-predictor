FROM python:3.11.0-slim

RUN pip install -U pip

# Purpose: Sets /app as the default working directory inside the container. All subsequent commands will be run from this directory.
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Create models directory and copy model
RUN mkdir -p models
COPY models/heart_disease_predictor.pkl models/

# Copy application code
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]