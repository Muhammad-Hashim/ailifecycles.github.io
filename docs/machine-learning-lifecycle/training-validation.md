# Training and Validation

## Introduction

Training and validation are the core phases where your model learns from data and demonstrates its ability to generalize. This phase involves feeding data through your chosen algorithm, optimizing parameters, and rigorously evaluating performance. Following Amazon's SageMaker practices, we'll explore comprehensive training strategies and validation techniques.

## Training Fundamentals

### Loss Functions
- **Regression**: Mean Squared Error (MSE), Mean Absolute Error (MAE)
- **Classification**: Cross-Entropy, Hinge Loss
- **Custom Losses**: Focal Loss for imbalanced data

### Optimization Algorithms
- **Gradient Descent**: Basic optimization
- **Stochastic Gradient Descent (SGD)**: Mini-batch updates
- **Adam**: Adaptive moment estimation
- **RMSProp**: Root mean square propagation

### Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

## Cross-Validation Techniques

### K-Fold Cross-Validation
```python
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    scores.append(accuracy_score(y_val, predictions))

print(f"Mean CV Score: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
```

### Stratified K-Fold
- Maintains class distribution in each fold
- Essential for imbalanced datasets

### Time Series Split
- Respects temporal order
- Prevents data leakage in time-dependent data

## Hyperparameter Tuning

### Manual Tuning
- Trial and error based on domain knowledge
- Time-consuming but insightful

### Automated Tuning
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
```

## Model Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Regression Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination

### Advanced Metrics
- **Confusion Matrix**: Detailed breakdown of predictions
- **Precision-Recall Curve**: For imbalanced datasets
- **Log Loss**: Probabilistic evaluation

## Handling Overfitting and Underfitting

### Overfitting Solutions
- **Regularization**: L1/L2 regularization
- **Dropout**: Randomly deactivating neurons
- **Early Stopping**: Stop training when validation loss increases
- **Data Augmentation**: Artificially increasing dataset size

### Underfitting Solutions
- **Increase Model Complexity**: More layers, parameters
- **Feature Engineering**: Better input features
- **Reduce Regularization**: Allow model to fit training data better

## Validation Strategies

### Holdout Validation
- Simple train/validation split
- Fast but may not be representative

### Cross-Validation
- More robust evaluation
- Computationally expensive

### Nested Cross-Validation
- For hyperparameter tuning and model selection
- Prevents overfitting during parameter optimization

## Training Best Practices

### Data Splitting
- **Train Set**: 60-80% for model learning
- **Validation Set**: 10-20% for hyperparameter tuning
- **Test Set**: 10-20% for final evaluation

### Batch Training
- **Batch Size**: Trade-off between speed and stability
- **Mini-batch Gradient Descent**: Most common approach

### Monitoring Training
- **Loss Curves**: Track training and validation loss
- **Learning Curves**: Plot performance vs. training size
- **Early Stopping**: Prevent overfitting

## Distributed Training

### Data Parallelism
- Splitting data across multiple GPUs/CPUs
- Synchronous vs. Asynchronous updates

### Model Parallelism
- Splitting large models across devices
- Pipeline parallelism

### Frameworks
- **Horovod**: Distributed training for TensorFlow/PyTorch
- **Ray**: Scalable distributed computing
- **AWS SageMaker Distributed Training**: Managed distributed training

## Transfer Learning

### Fine-tuning Pre-trained Models
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Freeze base layers
for param in model.base_model.parameters():
    param.requires_grad = False

# Fine-tune classifier layers
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
```

### Applications
- **Computer Vision**: Using ImageNet pre-trained models
- **NLP**: Fine-tuning BERT, GPT for specific tasks
- **Domain Adaptation**: Transferring knowledge across domains

## Resources

- [AWS SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)
- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [TensorFlow Training Guide](https://www.tensorflow.org/guide/keras/train_and_evaluate)
- [PyTorch Training Tutorials](https://pytorch.org/tutorials/beginner/basics/training_loop.html)
- [Hyperparameter Tuning Guide](https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide)
- [Cross-Validation Explained](https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85)
