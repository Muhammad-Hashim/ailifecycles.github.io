# Model Evaluation

## Overview

Model evaluation is a critical phase in the NLP lifecycle that determines how well your trained models perform on unseen data. This document covers comprehensive evaluation techniques, metrics, validation strategies, and error analysis methods for NLP models.

## Learning Objectives

By the end of this document, you will be able to:

- Understand different evaluation metrics for various NLP tasks
- Implement proper validation strategies
- Perform error analysis and model debugging
- Compare and select the best performing model
- Create evaluation pipelines for production deployment

## Evaluation Metrics

### Text Classification Metrics

For text classification tasks, use these primary metrics:

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_classification_model(model, X_test, y_test, class_names):
    """
    Comprehensive evaluation for classification models
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Per-class metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("\nPer-class Performance:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(".3f")
        print(".3f")
        print(".3f")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm
    }
```

### Named Entity Recognition (NER) Metrics

For NER tasks, use sequence-level metrics:

```python
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def evaluate_ner_model(model, X_test, y_test):
    """
    Evaluate NER model performance
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Overall metrics
    print("NER Evaluation Results:")
    print(classification_report(y_test, y_pred))

    # Entity-level F1 scores
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(".4f")
    print(".4f")
    print(".4f")

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

### Text Generation Metrics

For text generation tasks, use multiple metrics:

```python
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

def evaluate_generation_model(predictions, references):
    """
    Evaluate text generation model
    """
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    for pred, ref in zip(predictions, references):
        # BLEU score
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)

        # ROUGE scores
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # Average scores
    avg_bleu = np.mean(bleu_scores)
    avg_rouge = {key: np.mean(values) for key, values in rouge_scores.items()}

    print(".4f")
    for key, value in avg_rouge.items():
        print(".4f")

    return {
        'bleu': avg_bleu,
        'rouge': avg_rouge
    }
```

## Cross-Validation Strategies

### K-Fold Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def k_fold_cross_validation(model, X, y, k=5):
    """
    Perform k-fold cross-validation for NLP models
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_scores = {
        'accuracy': [],
        'f1_macro': [],
        'f1_weighted': []
    }

    fold = 1
    for train_idx, val_idx in skf.split(X, y):
        print(f"Fold {fold}/{k}")

        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Train model
        model.fit(X_train_fold, y_train_fold)

        # Predict
        y_pred_fold = model.predict(X_val_fold)

        # Calculate metrics
        acc = accuracy_score(y_val_fold, y_pred_fold)
        f1_macro = f1_score(y_val_fold, y_pred_fold, average='macro')
        f1_weighted = f1_score(y_val_fold, y_pred_fold, average='weighted')

        fold_scores['accuracy'].append(acc)
        fold_scores['f1_macro'].append(f1_macro)
        fold_scores['f1_weighted'].append(f1_weighted)

        print(".4f")
        print(".4f")
        print(".4f")

        fold += 1

    # Summary statistics
    print("\nCross-Validation Summary:")
    for metric, scores in fold_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(".4f")

    return fold_scores
```

### Time Series Cross-Validation

For temporal NLP data:

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cross_validation(model, X, y, n_splits=5):
    """
    Time series cross-validation for temporal NLP data
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        score = model.score(X_val, y_val)
        fold_scores.append(score)
        print(".4f")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(".4f")

    return fold_scores
```

## Error Analysis

### Confusion Matrix Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def analyze_errors(y_true, y_pred, class_names):
    """
    Perform detailed error analysis
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Error analysis
    print("\nError Analysis:")

    # Most confused pairs
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                error_rate = cm[i, j] / np.sum(cm[i, :])
                if error_rate > 0.1:  # Significant error rate
                    print(f"{class_names[i]} misclassified as {class_names[j]}: {cm[i, j]} times ({error_rate:.1%})")

    # Class-wise error rates
    print("\nClass-wise Error Rates:")
    for i, class_name in enumerate(class_names):
        total_samples = np.sum(cm[i, :])
        correct_predictions = cm[i, i]
        error_rate = 1 - (correct_predictions / total_samples)
        print(f"{class_name}: {error_rate:.1%} error rate")
```

### Prediction Confidence Analysis

```python
def analyze_prediction_confidence(model, X_test, y_test, class_names):
    """
    Analyze prediction confidence and uncertainty
    """
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_test)
        predictions = np.argmax(probas, axis=1)
        confidence = np.max(probas, axis=1)
    else:
        predictions = model.predict(X_test)
        confidence = np.ones(len(predictions))  # Assume confidence = 1 for non-probabilistic models

    # Analyze confidence distribution
    plt.figure(figsize=(12, 4))

    # Confidence histogram
    plt.subplot(1, 3, 1)
    plt.hist(confidence, bins=20, alpha=0.7)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')

    # Confidence vs correctness
    correct = (predictions == y_test)
    plt.subplot(1, 3, 2)
    plt.scatter(confidence, correct, alpha=0.6)
    plt.title('Confidence vs Correctness')
    plt.xlabel('Confidence')
    plt.ylabel('Correct (1) / Incorrect (0)')

    # Low confidence errors
    low_conf_mask = confidence < 0.5
    low_conf_correct = np.mean(correct[low_conf_mask]) if np.any(low_conf_mask) else 0
    high_conf_correct = np.mean(correct[~low_conf_mask]) if np.any(~low_conf_mask) else 0

    plt.subplot(1, 3, 3)
    plt.bar(['Low Confidence', 'High Confidence'],
            [low_conf_correct, high_conf_correct])
    plt.title('Accuracy by Confidence Level')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

    print(".4f")
    print(".4f")

    return confidence, predictions
```

## Model Comparison and Selection

### Model Comparison Framework

```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compare_models(models, X_test, y_test, model_names):
    """
    Compare multiple models comprehensively
    """
    results = []

    for model, name in zip(models, model_names):
        print(f"Evaluating {name}...")

        # Get predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Training time (if available)
        training_time = getattr(model, 'training_time_', None)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time': training_time
        })

    # Create comparison table
    df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(df.to_string(index=False))

    # Rank models by F1-score
    df_sorted = df.sort_values('F1-Score', ascending=False)
    print("
Models ranked by F1-Score:")
    for idx, row in df_sorted.iterrows():
        print(f"{idx+1}. {row['Model']}: {row['F1-Score']:.4f}")

    return df
```

### Statistical Significance Testing

```python
from scipy import stats
import numpy as np

def statistical_significance_test(scores1, scores2, alpha=0.05):
    """
    Test if performance difference between two models is statistically significant
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        print("The performance difference is statistically significant.")
    else:
        print("The performance difference is not statistically significant.")

    return p_value < alpha
```

## Evaluation Best Practices

### 1. Multiple Metrics

- Use task-appropriate metrics
- Consider business requirements
- Don't rely on single metrics

### 2. Cross-Validation

- Use stratified k-fold for imbalanced data
- Consider time series split for temporal data
- Report mean and standard deviation

### 3. Error Analysis

- Analyze confusion matrices
- Examine prediction confidence
- Identify systematic errors

### 4. Statistical Testing

- Test significance of differences
- Use appropriate statistical tests
- Consider practical significance

### 5. Robust Evaluation

- Evaluate on multiple datasets
- Test model robustness
- Consider edge cases

## Common Pitfalls

1. **Data Leakage**: Ensure no test data in training
2. **Metric Selection**: Choose metrics matching business goals
3. **Overfitting to Validation**: Don't tune hyperparameters on test set
4. **Ignoring Class Imbalance**: Use appropriate metrics for imbalanced data
5. **Single Metric Focus**: Consider multiple evaluation aspects

## Next Steps

After evaluating your models:

1. **Model Deployment**: Deploy best performing model
2. **Monitoring**: Set up production monitoring
3. **Maintenance**: Plan for model updates and retraining
4. **Documentation**: Document evaluation results and decisions

Continue to [Model Deployment](./model-deployment.md) to learn how to deploy your evaluated models to production.

## Resources

- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/index)
- [Papers with Code: Evaluation Metrics](https://paperswithcode.com/task/model-evaluation)
- [Google Research: Evaluation Metrics](https://research.google/tools/datasets/)
