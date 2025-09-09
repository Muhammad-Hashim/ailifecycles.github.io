# Model Deployment

## Overview

Model deployment is the final phase of the NLP lifecycle where trained and evaluated models are made available for production use. This document covers deployment strategies, infrastructure considerations, monitoring, and maintenance practices for NLP models in production environments.

## Learning Objectives

By the end of this document, you will be able to:

- Understand different deployment architectures for NLP models
- Implement model serving solutions
- Set up monitoring and logging for production models
- Handle model updates and versioning
- Ensure scalability and performance optimization

## Deployment Strategies

### 1. REST API Deployment

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
from typing import Dict, Any

class NLPModelService:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.app = Flask(__name__)
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy'})

        @self.app.route('/predict', methods=['POST'])
        def predict():
            return self._predict_endpoint()

    def _predict_endpoint(self):
        try:
            data = request.get_json()

            if 'text' not in data:
                return jsonify({'error': 'Missing text field'}), 400

            text = data['text']

            # Preprocess text
            text_vectorized = self.vectorizer.transform([text])

            # Make prediction
            prediction = self.model.predict(text_vectorized)
            probabilities = self.model.predict_proba(text_vectorized)

            # Get prediction confidence
            confidence = np.max(probabilities[0])

            response = {
                'prediction': prediction[0],
                'confidence': float(confidence),
                'probabilities': probabilities[0].tolist()
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port, debug=False)

# Usage
if __name__ == '__main__':
    service = NLPModelService('model.pkl', 'vectorizer.pkl')
    service.run()
```

### 2. FastAPI Deployment

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Optional
import uvicorn

app = FastAPI(title="NLP Model API", version="1.0.0")

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

class PredictionRequest(BaseModel):
    text: str
    return_probabilities: Optional[bool] = False

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Optional[List[float]] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Vectorize text
        text_vectorized = vectorizer.transform([request.text])

        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = float(np.max(probabilities))

        response = PredictionResponse(
            prediction=prediction,
            confidence=confidence
        )

        if request.return_probabilities:
            response.probabilities = probabilities.tolist()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(texts: List[str]):
    try:
        # Vectorize texts
        texts_vectorized = vectorizer.transform(texts)

        # Make predictions
        predictions = model.predict(texts_vectorized)
        probabilities = model.predict_proba(texts_vectorized)

        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                'text_index': i,
                'prediction': pred,
                'confidence': float(np.max(probs)),
                'probabilities': probs.tolist()
            })

        return {'results': results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Docker Containerization

```dockerfile
# Dockerfile for NLP Model Service
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and application code
COPY model.pkl vectorizer.pkl ./
COPY app.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  nlp-model:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.pkl
      - VECTORIZER_PATH=/app/vectorizer.pkl
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - nlp-model
```

## Model Serving with MLflow

```python
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd

def deploy_with_mlflow(model, vectorizer, X_train, y_train):
    """
    Deploy model using MLflow
    """
    # Set experiment
    mlflow.set_experiment("NLP_Model_Deployment")

    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("vectorizer_type", type(vectorizer).__name__)

        # Create model signature
        sample_input = X_train[:5]
        sample_output = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        # Log the model
        mlflow.sklearn.log_model(
            model,
            "nlp_model",
            signature=signature,
            input_example=sample_input
        )

        # Log vectorizer as artifact
        import joblib
        joblib.dump(vectorizer, "vectorizer.pkl")
        mlflow.log_artifact("vectorizer.pkl", "vectorizer")

        print(f"Model logged with run ID: {mlflow.active_run().info.run_id}")

def load_mlflow_model(run_id: str):
    """
    Load model from MLflow
    """
    model_uri = f"runs:/{run_id}/nlp_model"
    model = mlflow.sklearn.load_model(model_uri)

    # Load vectorizer
    client = mlflow.tracking.MlflowClient()
    vectorizer_path = client.download_artifacts(run_id, "vectorizer/vectorizer.pkl")
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer
```

## Scalability and Performance

### Load Balancing

```python
# Gunicorn configuration for scaling
# gunicorn_config.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
```

### Model Optimization

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort
import numpy as np

def optimize_transformer_model(model_name: str, output_path: str):
    """
    Optimize transformer model for production
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Convert to ONNX
    dummy_input = tokenizer("Hello world", return_tensors="pt")

    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

    print(f"Model optimized and saved to {output_path}")

def load_optimized_model(model_path: str):
    """
    Load optimized ONNX model
    """
    session = ort.InferenceSession(model_path)
    return session

def predict_with_onnx(session, tokenizer, text: str):
    """
    Make prediction with optimized model
    """
    inputs = tokenizer(text, return_tensors="np")

    # Run inference
    outputs = session.run(
        None,
        {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
    )

    logits = outputs[0]
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()

    return probabilities
```

## Monitoring and Logging

### Production Monitoring

```python
import logging
import time
from functools import wraps
import psutil
import GPUtil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_service.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def monitor_performance(func):
    """
    Decorator to monitor function performance
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log performance metrics
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")

            # Monitor system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            logger.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_percent}%")

            # GPU monitoring (if available)
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    logger.info(f"GPU {gpu.id}: {gpu.load*100:.1f}% used, {gpu.memoryUsed}MB used")
            except:
                pass

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {str(e)}")
            raise

    return wrapper

class ModelMonitor:
    def __init__(self):
        self.prediction_count = 0
        self.error_count = 0
        self.response_times = []

    def log_prediction(self, response_time: float, success: bool):
        self.prediction_count += 1
        self.response_times.append(response_time)

        if not success:
            self.error_count += 1

        # Log metrics every 100 predictions
        if self.prediction_count % 100 == 0:
            self._log_metrics()

    def _log_metrics(self):
        avg_response_time = np.mean(self.response_times[-100:])
        error_rate = self.error_count / self.prediction_count

        logger.info(f"Predictions: {self.prediction_count}")
        logger.info(f"Average response time: {avg_response_time:.4f}s")
        logger.info(f"Error rate: {error_rate:.4f}")

# Global monitor instance
monitor = ModelMonitor()
```

### Health Checks and Alerts

```python
import requests
import time
from typing import Dict, Any

class HealthChecker:
    def __init__(self, service_url: str, check_interval: int = 60):
        self.service_url = service_url
        self.check_interval = check_interval
        self.last_check = 0
        self.is_healthy = True

    def check_health(self) -> Dict[str, Any]:
        """
        Perform health check on the service
        """
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            is_healthy = response.status_code == 200

            result = {
                'healthy': is_healthy,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'timestamp': time.time()
            }

            if not is_healthy:
                logger.warning(f"Health check failed: {result}")

            return result

        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }

    def monitor_service(self):
        """
        Continuously monitor service health
        """
        while True:
            current_time = time.time()

            if current_time - self.last_check >= self.check_interval:
                health_status = self.check_health()
                self.last_check = current_time

                if health_status['healthy'] != self.is_healthy:
                    if health_status['healthy']:
                        logger.info("Service is back to healthy state")
                    else:
                        logger.error("Service health check failed")
                        self._send_alert(health_status)

                self.is_healthy = health_status['healthy']

            time.sleep(10)  # Check every 10 seconds

    def _send_alert(self, health_status: Dict[str, Any]):
        """
        Send alert when service becomes unhealthy
        """
        # Implement your alerting mechanism here
        # Could be email, Slack, PagerDuty, etc.
        logger.critical(f"ALERT: Service is unhealthy - {health_status}")
```

## Model Versioning and Updates

### Model Registry

```python
import json
import os
from datetime import datetime
from typing import Dict, Any, List

class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {'models': {}, 'active_model': None}

    def _save_registry(self):
        """Save model registry to file"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_model(self, model_id: str, model_path: str, metadata: Dict[str, Any]):
        """Register a new model version"""
        model_info = {
            'model_id': model_id,
            'model_path': model_path,
            'metadata': metadata,
            'registered_at': datetime.now(),
            'performance_metrics': {},
            'is_active': False
        }

        self.registry['models'][model_id] = model_info
        self._save_registry()

        logger.info(f"Model {model_id} registered successfully")

    def activate_model(self, model_id: str):
        """Activate a model for production use"""
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")

        # Deactivate current active model
        if self.registry['active_model']:
            self.registry['models'][self.registry['active_model']]['is_active'] = False

        # Activate new model
        self.registry['models'][model_id]['is_active'] = True
        self.registry['active_model'] = model_id
        self._save_registry()

        logger.info(f"Model {model_id} activated for production")

    def get_active_model(self) -> Dict[str, Any]:
        """Get currently active model"""
        if not self.registry['active_model']:
            raise ValueError("No active model found")

        return self.registry['models'][self.registry['active_model']]

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return list(self.registry['models'].values())

    def update_model_metrics(self, model_id: str, metrics: Dict[str, Any]):
        """Update performance metrics for a model"""
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")

        self.registry['models'][model_id]['performance_metrics'] = metrics
        self._save_registry()

        logger.info(f"Metrics updated for model {model_id}")
```

## A/B Testing Framework

```python
import random
import hashlib
from typing import Dict, Any, List
import numpy as np

class ABTestingFramework:
    def __init__(self, models: Dict[str, Any], traffic_distribution: Dict[str, float]):
        """
        Initialize A/B testing with multiple models

        Args:
            models: Dictionary of model_id -> model_instance
            traffic_distribution: Dictionary of model_id -> traffic_percentage
        """
        self.models = models
        self.traffic_distribution = traffic_distribution
        self.validate_distribution()

        # Tracking metrics
        self.metrics = {model_id: {'requests': 0, 'correct': 0} for model_id in models.keys()}

    def validate_distribution(self):
        """Validate that traffic distribution sums to 100%"""
        total = sum(self.traffic_distribution.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Traffic distribution must sum to 1.0, got {total}")

    def get_model_for_request(self, user_id: str) -> str:
        """
        Determine which model to use for a request based on user ID
        """
        # Create hash of user ID for consistent routing
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        random.seed(hash_value)

        # Select model based on traffic distribution
        rand_val = random.random()
        cumulative = 0.0

        for model_id, percentage in self.traffic_distribution.items():
            cumulative += percentage
            if rand_val <= cumulative:
                return model_id

        # Fallback (should not happen with proper distribution)
        return list(self.models.keys())[0]

    def make_prediction(self, user_id: str, text: str) -> Dict[str, Any]:
        """
        Make prediction using A/B testing
        """
        model_id = self.get_model_for_request(user_id)
        model = self.models[model_id]

        # Make prediction
        prediction = model.predict(text)

        # Track metrics (in real implementation, you'd track ground truth)
        self.metrics[model_id]['requests'] += 1

        return {
            'model_id': model_id,
            'prediction': prediction,
            'user_id': user_id
        }

    def get_ab_test_results(self) -> Dict[str, Any]:
        """
        Get current A/B test results
        """
        results = {}

        for model_id, metrics in self.metrics.items():
            requests = metrics['requests']
            if requests > 0:
                accuracy = metrics['correct'] / requests
            else:
                accuracy = 0.0

            results[model_id] = {
                'requests': requests,
                'accuracy': accuracy,
                'traffic_percentage': self.traffic_distribution.get(model_id, 0)
            }

        return results

    def update_ground_truth(self, user_id: str, prediction: str, actual: str):
        """
        Update metrics with ground truth
        """
        model_id = self.get_model_for_request(user_id)

        if prediction == actual:
            self.metrics[model_id]['correct'] += 1
```

## Security Considerations

### Input Validation and Sanitization

```python
import re
from typing import Optional

class InputValidator:
    def __init__(self, max_length: int = 10000, allowed_chars: Optional[str] = None):
        self.max_length = max_length
        self.allowed_chars = allowed_chars or r"[^a-zA-Z0-9\s\.,!?\-\'\"]"

    def validate_text(self, text: str) -> tuple[bool, str]:
        """
        Validate input text for security and quality
        """
        # Check length
        if len(text) > self.max_length:
            return False, f"Text too long: {len(text)} > {self.max_length}"

        # Check for potentially harmful content
        if self._contains_malicious_content(text):
            return False, "Potentially malicious content detected"

        # Sanitize text
        sanitized_text = self._sanitize_text(text)

        return True, sanitized_text

    def _contains_malicious_content(self, text: str) -> bool:
        """
        Check for malicious content patterns
        """
        malicious_patterns = [
            r"<script[^>]*>.*?</script>",  # Script tags
            r"javascript:",                # JavaScript URLs
            r"on\w+\s*=",                  # Event handlers
            r"eval\s*\(",                  # Eval functions
        ]

        for pattern in malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text by removing or escaping potentially harmful content
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove excessive whitespace
        text = ' '.join(text.split())

        return text
```

## Deployment Best Practices

### 1. Containerization

- Use Docker for consistent environments
- Implement multi-stage builds for optimization
- Use appropriate base images

### 2. Scalability

- Implement horizontal scaling
- Use load balancers
- Monitor resource usage

### 3. Monitoring

- Track prediction latency and throughput
- Monitor model performance drift
- Set up alerting for anomalies

### 4. Security

- Validate all inputs
- Implement rate limiting
- Use HTTPS in production

### 5. Versioning

- Maintain model version history
- Implement gradual rollouts
- Support rollback capabilities

## Common Deployment Challenges

1. **Cold Start Latency**: Pre-warm models or use model caching
2. **Memory Constraints**: Optimize model size and batch processing
3. **Model Drift**: Implement continuous monitoring and retraining
4. **Scalability**: Use serverless or container orchestration
5. **Cost Optimization**: Balance performance with infrastructure costs

## Next Steps

After deploying your NLP models:

1. **Monitoring Setup**: Implement comprehensive monitoring
2. **Performance Optimization**: Continuously optimize for latency and cost
3. **Model Maintenance**: Plan for regular model updates and retraining
4. **Documentation**: Document deployment procedures and APIs

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [MLflow Model Serving](https://mlflow.org/docs/latest/models.html)
- [Kubernetes for ML](https://www.kubeflow.org/)
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [Google Cloud AI Platform](https://cloud.google.com/ai-platform)
