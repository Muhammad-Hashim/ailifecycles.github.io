# Model Training

Model training is the process of teaching machine learning models to make predictions or classifications based on your prepared data. This phase involves selecting appropriate algorithms, configuring hyperparameters, and training models to achieve optimal performance.

## Model Selection

### Traditional Machine Learning Models

Start with interpretable baseline models:

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Prepare data
texts = df['processed_text'].tolist()
labels = df['label'].tolist()  # Your target labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train baseline models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train_vec, y_train)

    # Make predictions
    y_pred = model.predict(X_test_vec)

    # Evaluate
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))

    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': (y_pred == y_test).mean()
    }
```

### Deep Learning Models

Use neural networks for complex patterns:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Prepare data for deep learning
max_words = 10000
max_len = 200

# Tokenization
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Convert labels to categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

y_train_cat = tf.keras.utils.to_categorical(y_train_enc)
y_test_cat = tf.keras.utils.to_categorical(y_test_enc)

# Build LSTM model
def build_lstm_model(vocab_size, embedding_dim=128, max_len=200):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Train LSTM model
lstm_model = build_lstm_model(max_words)
print(lstm_model.summary())

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_lstm_model.h5', monitor='val_accuracy', save_best_only=True)
]

# Train
history = lstm_model.fit(
    X_train_pad, y_train_cat,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

# Evaluate
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_pad, y_test_cat)
print(f"LSTM Test Accuracy: {lstm_accuracy:.4f}")
```

## Transformer Models

### BERT and Modern Transformers

Use pre-trained transformer models:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load pre-trained model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(set(labels))
)

# Prepare datasets
train_dataset = TextDataset(X_train, y_train_enc, tokenizer)
test_dataset = TextDataset(X_test, y_test_enc, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda pred: {
        'accuracy': (pred.predictions.argmax(-1) == pred.label_ids).mean()
    }
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"BERT Test Results: {results}")
```

## Hyperparameter Tuning

### Grid Search and Random Search

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

# Example with SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'loss': ['hinge', 'squared_hinge']
}

# Grid search
grid_search = GridSearchCV(
    LinearSVC(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train_vec, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Random search for larger parameter space
param_dist = {
    'C': uniform(0.1, 100),
    'max_iter': randint(1000, 5000)
}

random_search = RandomizedSearchCV(
    LinearSVC(random_state=42),
    param_dist,
    n_iter=20,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_vec, y_train)
print(f"Random search best parameters: {random_search.best_params_}")
print(f"Random search best score: {random_search.best_score_:.4f}")
```

### Bayesian Optimization

```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Define search space
search_spaces = {
    'C': Real(0.1, 100, prior='log-uniform'),
    'max_iter': Integer(1000, 5000),
    'penalty': Categorical(['l1', 'l2']),
    'loss': Categorical(['hinge', 'squared_hinge'])
}

# Bayesian optimization
bayes_search = BayesSearchCV(
    LinearSVC(random_state=42),
    search_spaces,
    n_iter=32,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train_vec, y_train)
print(f"Bayesian optimization best parameters: {bayes_search.best_params_}")
print(f"Bayesian optimization best score: {bayes_search.best_score_:.4f}")
```

## Handling Class Imbalance

### Resampling Techniques

```python
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Check class distribution
print("Class distribution:")
print(pd.Series(y_train).value_counts())

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)

print("After SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Random oversampling
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train_vec, y_train)

# Random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_vec, y_train)

# Train model with balanced data
model_balanced = LogisticRegression(random_state=42, max_iter=1000)
model_balanced.fit(X_train_smote, y_train_smote)

# Evaluate
y_pred_balanced = model_balanced.predict(X_test_vec)
print("Balanced model results:")
print(classification_report(y_test, y_pred_balanced))
```

### Class Weighting

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("Class weights:", class_weight_dict)

# Train with class weights
model_weighted = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight=class_weight_dict
)

model_weighted.fit(X_train_vec, y_train)

# Evaluate
y_pred_weighted = model_weighted.predict(X_test_vec)
print("Weighted model results:")
print(classification_report(y_test, y_pred_weighted))
```

## Cross-Validation Strategies

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Stratified K-Fold for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
model = LogisticRegression(random_state=42, max_iter=1000)
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=skf, scoring='f1_macro')

print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Detailed cross-validation
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(model, X_train_vec, y_train, cv=skf, scoring=scoring)

print("Detailed CV results:")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## Model Interpretability

### Feature Importance Analysis

```python
# For linear models
if hasattr(model, 'coef_'):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    # Get top positive and negative features
    top_positive = coefficients.argsort()[-10:][::-1]
    top_negative = coefficients.argsort()[:10]

    print("Top positive features:")
    for idx in top_positive:
        print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")

    print("\nTop negative features:")
    for idx in top_negative:
        print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")

# For tree-based models
if hasattr(model, 'feature_importances_'):
    feature_importance = model.feature_importances_
    top_features = feature_importance.argsort()[-10:][::-1]

    print("Top important features:")
    for idx in top_features:
        print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")
```

## Training Best Practices

### Data Preparation

1. **Train/Validation/Test Split**: Always use separate datasets
2. **Stratification**: Maintain class distribution across splits
3. **Preprocessing Consistency**: Apply same preprocessing to all splits
4. **Data Leakage Prevention**: Avoid information leakage between splits

5. **Start Simple**: Begin with baseline models
6. **Progressive Complexity**: Gradually increase model complexity
7. **Early Stopping**: Prevent overfitting with early stopping
8. **Regularization**: Use appropriate regularization techniques
9. **Cross-Validation**: Validate on multiple data folds

### Performance Monitoring

1. **Track Metrics**: Monitor relevant performance metrics
2. **Learning Curves**: Analyze training vs validation performance
3. **Overfitting Detection**: Watch for training/validation divergence
4. **Computational Resources**: Monitor memory and time usage

## Next Steps

After training your models:

1. **Model Evaluation**: Comprehensive evaluation on test data
2. **Model Comparison**: Compare different model performances
3. **Error Analysis**: Analyze model mistakes and failure modes
4. **Model Selection**: Choose best model for deployment
5. **Hyperparameter Optimization**: Fine-tune selected model

Continue to [Model Evaluation](./model-evaluation.md) to learn comprehensive evaluation techniques for your trained models.
