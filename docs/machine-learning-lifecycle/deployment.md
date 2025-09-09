# Deployment

## Introduction

Deployment is the phase where your trained model moves from development to production, becoming accessible to end-users or integrated into applications. This critical phase involves considerations of scalability, latency, reliability, and maintenance. Amazon's SageMaker provides excellent guidance on model deployment, which we'll follow here.

## Deployment Strategies

### Batch Inference
- **When to Use**: Non-real-time predictions, large datasets
- **Advantages**: Cost-effective, handles large volumes
- **Tools**: AWS Batch, Apache Spark

### Real-time Inference
- **When to Use**: Immediate predictions required
- **Challenges**: Latency, throughput, availability
- **Architectures**: REST APIs, gRPC services

### Edge Deployment
- **When to Use**: Low-latency requirements, offline capabilities
- **Platforms**: Mobile devices, IoT, embedded systems
- **Frameworks**: TensorFlow Lite, ONNX Runtime

## Model Serving Frameworks

### TensorFlow Serving
```python
# Export model
model.save('path/to/model')

# Serve with TensorFlow Serving
# Command: tensorflow_model_server --rest_api_port=8501 --model_name=my_model --model_base_path=/path/to/model
```

### TorchServe
```python
# Create model archive
torch-model-archiver --model-name my_model --version 1.0 --model-file model.py --serialized-file model.pth --handler handler.py

# Start server
torchserve --start --model-store model_store --models my_model.mar
```

### FastAPI with ML Models
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])
    return {"prediction": prediction.tolist()}
```

## Containerization and Orchestration

### Docker for ML Models
```dockerfile
FROM python:3.8-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: my-ml-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## Cloud Deployment Options

### AWS SageMaker
- **Endpoints**: Real-time inference
- **Batch Transform**: Batch processing
- **Serverless Inference**: Auto-scaling
- **Multi-model Endpoints**: Cost optimization

### Google Cloud AI Platform
- **AI Platform Prediction**: Managed model serving
- **Vertex AI**: Unified ML platform

### Azure Machine Learning
- **Real-time Endpoints**: Low-latency predictions
- **Batch Endpoints**: Asynchronous processing

## Model Optimization for Production

### Model Compression
- **Quantization**: Reducing precision (float32 → int8)
- **Pruning**: Removing unnecessary weights
- **Knowledge Distillation**: Training smaller models

### Performance Optimization
```python
# ONNX conversion for cross-platform deployment
import torch
from torch.onnxruntime import InferenceSession

# Convert PyTorch model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Load and run inference
session = InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_data})
```

## Monitoring and Logging

### Application Metrics
- **Latency**: Response time tracking
- **Throughput**: Requests per second
- **Error Rate**: Failed predictions percentage

### Model Metrics
- **Data Drift**: Input distribution changes
- **Model Drift**: Prediction distribution changes
- **Performance Degradation**: Accuracy over time

### Logging Frameworks
- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Prometheus + Grafana**: Metrics collection and visualization
- **AWS CloudWatch**: Cloud-native monitoring

## A/B Testing and Canary Deployments

### A/B Testing
```python
import numpy as np

def ab_test(model_a, model_b, test_data, alpha=0.05):
    predictions_a = model_a.predict(test_data)
    predictions_b = model_b.predict(test_data)
    
    # Statistical significance test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(predictions_a, predictions_b)
    
    return p_value < alpha
```

### Canary Deployment
- Roll out new model to small percentage of traffic
- Monitor performance before full deployment
- Automated rollback if issues detected

## Security and Compliance

### Model Security
- **Input Validation**: Sanitize inputs to prevent attacks
- **Rate Limiting**: Prevent abuse
- **Authentication**: Secure API access

### Data Privacy
- **Encryption**: Protect data in transit and at rest
- **Anonymization**: Remove sensitive information
- **Compliance**: GDPR, HIPAA, industry regulations

## CI/CD for ML

### ML Pipelines
- **GitOps**: Version control for models and code
- **Automated Testing**: Unit tests, integration tests
- **Continuous Deployment**: Automated model updates

### Tools
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data version control
- **GitHub Actions**: CI/CD for ML projects

## Scaling Strategies

### Horizontal Scaling
- Load balancers distributing traffic
- Auto-scaling groups based on metrics

### Vertical Scaling
- Increasing instance size for compute-intensive models
- GPU acceleration for deep learning models

### Serverless Options
- AWS Lambda for lightweight models
- Google Cloud Functions
- Azure Functions

## Resources

- [AWS SageMaker Deployment](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [TensorFlow Serving Guide](https://www.tensorflow.org/tfx/guide/serving)
- [TorchServe Documentation](https://pytorch.org/serve/)
- [Kubernetes for ML](https://www.kubeflow.org/docs/)
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- [Production ML Book](https://www.manning.com/books/building-machine-learning-pipelines)
