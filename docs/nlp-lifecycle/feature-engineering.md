# Feature Engineering

Feature engineering is the process of transforming raw text data into numerical representations that machine learning models can understand and learn from. This crucial step bridges the gap between human-readable text and mathematical models.

## Traditional Text Features

### Bag of Words (BoW)

Convert text to vectors based on word frequency:

```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample data
texts = [
    "I love machine learning",
    "Machine learning is amazing",
    "I enjoy deep learning",
    "Deep learning and machine learning are related"
]

# Create Bag of Words features
vectorizer = CountVectorizer()
bow_features = vectorizer.fit_transform(texts)

# Convert to DataFrame for visualization
feature_names = vectorizer.get_feature_names_out()
bow_df = pd.DataFrame(bow_features.toarray(), columns=feature_names)

print("Vocabulary:", feature_names)
print("\nBag of Words Matrix:")
print(bow_df)

# Feature analysis
print(f"\nTotal unique words: {len(feature_names)}")
print(f"Sparsity: {bow_features.nnz / (bow_features.shape[0] * bow_features.shape[1]):.3f}")
```

### TF-IDF (Term Frequency-Inverse Document Frequency)

Weight words by importance across the corpus:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(texts)

# Convert to DataFrame
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("TF-IDF Features:")
print(tfidf_df)

# Analyze TF-IDF scores
print("\nTop TF-IDF scores for first document:")
first_doc_tfidf = tfidf_features[0].toarray()[0]
top_indices = first_doc_tfidf.argsort()[-5:][::-1]  # Top 5
for idx in top_indices:
    word = tfidf_vectorizer.get_feature_names_out()[idx]
    score = first_doc_tfidf[idx]
    print(f"{word}: {score:.3f}")
```

### N-gram Features

Capture word sequences and context:

```python
# Unigrams, bigrams, and trigrams
ngram_vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
ngram_features = ngram_vectorizer.fit_transform(texts)

print("N-gram features shape:", ngram_features.shape)
print("Sample n-grams:")
feature_names = ngram_vectorizer.get_feature_names_out()
for i, feature in enumerate(feature_names[:10]):
    print(f"{i+1}. {feature}")
```

## Word Embeddings

### Word2Vec

Train word embeddings on your corpus:

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download punkt for tokenization
nltk.download('punkt')

# Prepare tokenized sentences
tokenized_texts = [word_tokenize(text.lower()) for text in texts]

# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=100,  # Embedding dimension
    window=5,         # Context window size
    min_count=1,      # Minimum word frequency
    workers=4,        # Number of worker threads
    epochs=10         # Number of training epochs
)

# Get word vectors
word_vectors = w2v_model.wv

print("Vocabulary size:", len(word_vectors))
print("Embedding dimension:", word_vectors.vector_size)

# Find similar words
if 'learning' in word_vectors:
    similar_words = word_vectors.most_similar('learning', topn=5)
    print("\nWords similar to 'learning':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.3f}")

# Get document vectors (average of word vectors)
def get_doc_vector(text, model):
    tokens = word_tokenize(text.lower())
    vectors = [model.wv[word] for word in tokens if word in model.wv]

    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return [0] * model.vector_size

# Create document embeddings
doc_embeddings = [get_doc_vector(text, w2v_model) for text in texts]
print(f"\nDocument embeddings shape: {len(doc_embeddings)} x {len(doc_embeddings[0])}")
```

### GloVe Embeddings

Load pre-trained GloVe embeddings:

```python
import numpy as np

def load_glove_embeddings(file_path):
    """
    Load GloVe embeddings from file
    """
    embeddings_index = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index

# Load GloVe (assuming you have the file)
# glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')

# For demonstration, create a simple embedding matrix
vocab = set()
for text in tokenized_texts:
    vocab.update(text)

vocab_size = len(vocab)
embedding_dim = 100

# Create random embeddings (replace with actual GloVe)
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

print(f"Embedding matrix shape: {embedding_matrix.shape}")
```

## Contextual Embeddings

### BERT Embeddings

Use contextual embeddings from transformer models:

```python
from transformers import AutoTokenizer, AutoModel
import torch

def get_bert_embeddings(texts, model_name='bert-base-uncased'):
    """
    Generate BERT embeddings for texts
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []

    for text in texts:
        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                          padding=True, max_length=512)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding as document representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        embeddings.append(cls_embedding.numpy())

    return np.array(embeddings)

# Generate BERT embeddings
bert_embeddings = get_bert_embeddings(texts)
print(f"BERT embeddings shape: {bert_embeddings.shape}")
```

### Sentence Transformers

Use sentence-level embeddings:

```python
from sentence_transformers import SentenceTransformer

def get_sentence_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Generate sentence-level embeddings
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)

    return embeddings

# Generate sentence embeddings
sentence_embeddings = get_sentence_embeddings(texts)
print(f"Sentence embeddings shape: {sentence_embeddings.shape}")

# Calculate similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(sentence_embeddings)
print("\nSentence similarity matrix:")
print(similarity_matrix)
```

## Advanced Feature Engineering

### Topic Modeling Features

Extract topic-based features:

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def extract_topic_features(texts, n_topics=5):
    """
    Extract topic modeling features
    """
    # Vectorize texts
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    dtm = vectorizer.fit_transform(texts)

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_distributions = lda.fit_transform(dtm)

    return topic_distributions, lda, vectorizer

# Extract topic features
topic_features, lda_model, vectorizer = extract_topic_features(texts)
print(f"Topic features shape: {topic_features.shape}")

# Show top words for each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-5:-1]]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
```

### Syntactic Features

Extract syntactic and grammatical features:

```python
import spacy

def extract_syntactic_features(texts):
    """
    Extract syntactic features using spaCy
    """
    nlp = spacy.load('en_core_web_sm')

    features = []

    for text in texts:
        doc = nlp(text)

        # Extract features
        feature_dict = {
            'num_sentences': len(list(doc.sents)),
            'num_tokens': len(doc),
            'num_words': sum(1 for token in doc if token.is_alpha),
            'num_nouns': sum(1 for token in doc if token.pos_ == 'NOUN'),
            'num_verbs': sum(1 for token in doc if token.pos_ == 'VERB'),
            'num_adjectives': sum(1 for token in doc if token.pos_ == 'ADJ'),
            'avg_word_length': sum(len(token) for token in doc if token.is_alpha) / max(1, sum(1 for token in doc if token.is_alpha)),
            'lexical_diversity': len(set(token.lemma_.lower() for token in doc if token.is_alpha)) / max(1, sum(1 for token in doc if token.is_alpha))
        }

        features.append(feature_dict)

    return pd.DataFrame(features)

# Extract syntactic features
syntactic_features = extract_syntactic_features(texts)
print("Syntactic features:")
print(syntactic_features)
```

### Readability Features

Extract readability and complexity metrics:

```python
import textstat

def extract_readability_features(texts):
    """
    Extract readability features
    """
    features = []

    for text in texts:
        feature_dict = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'smog_index': textstat.smog_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
            'difficult_words': textstat.difficult_words(text),
            'linsear_write_formula': textstat.linsear_write_formula(text),
            'gunning_fog': textstat.gunning_fog(text)
        }

        features.append(feature_dict)

    return pd.DataFrame(features)

# Extract readability features
readability_features = extract_readability_features(texts)
print("Readability features:")
print(readability_features)
```

## Feature Selection and Dimensionality Reduction

### Feature Selection

Select most important features:

```python
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Example with classification labels
labels = ['positive', 'positive', 'positive', 'neutral']  # Example labels

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Feature selection with TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf_vectorizer.fit_transform(texts)

# Select top k features
selector = SelectKBest(score_func=chi2, k=50)
selected_features = selector.fit_transform(tfidf_features, encoded_labels)

print(f"Original features: {tfidf_features.shape[1]}")
print(f"Selected features: {selected_features.shape[1]}")

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_feature_names = [tfidf_vectorizer.get_feature_names_out()[i] for i in selected_indices]
print("Top selected features:", selected_feature_names[:10])
```

### Dimensionality Reduction

Reduce feature dimensions:

```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

def reduce_dimensions(features, method='pca', n_components=50):
    """
    Reduce dimensionality of features
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)  # t-SNE typically 2D
    else:
        raise ValueError("Invalid reduction method")

    reduced_features = reducer.fit_transform(features.toarray() if hasattr(features, 'toarray') else features)

    return reduced_features, reducer

# Reduce TF-IDF dimensions
reduced_features, pca_model = reduce_dimensions(tfidf_features, method='pca', n_components=10)
print(f"Reduced features shape: {reduced_features.shape}")
print(f"Explained variance ratio: {pca_model.explained_variance_ratio_}")
```

## Feature Engineering Pipeline

### Complete Feature Engineering Function

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for text feature extraction
    """

    def __init__(self, method='tfidf', embedding_model=None):
        self.method = method
        self.embedding_model = embedding_model
        self.vectorizer = None

    def fit(self, X, y=None):
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.vectorizer.fit(X)
        elif self.method == 'bow':
            self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
            self.vectorizer.fit(X)
        # Add other methods as needed

        return self

    def transform(self, X):
        if self.method in ['tfidf', 'bow']:
            return self.vectorizer.transform(X)
        # Add other transformation logic

def create_feature_pipeline():
    """
    Create a complete feature engineering pipeline
    """
    # Text feature extraction
    text_features = Pipeline([
        ('extractor', TextFeatureExtractor(method='tfidf')),
        ('scaler', StandardScaler(with_mean=False))  # For sparse matrices
    ])

    # Additional feature extraction (syntactic, readability)
    additional_features = FeatureUnion([
        ('syntactic', extract_syntactic_features),
        ('readability', extract_readability_features)
    ])

    # Combine all features
    feature_pipeline = FeatureUnion([
        ('text_features', text_features),
        ('additional_features', additional_features)
    ])

    return feature_pipeline

# Create and use pipeline
pipeline = create_feature_pipeline()
X_features = pipeline.fit_transform(texts)
print(f"Final feature matrix shape: {X_features.shape}")
```

## Best Practices

### Feature Engineering Guidelines

1. **Understand Your Data**: Analyze text characteristics before feature engineering
2. **Start Simple**: Begin with basic features and add complexity gradually
3. **Domain Knowledge**: Incorporate domain-specific features when available
4. **Computational Efficiency**: Balance feature richness with computational cost
5. **Validation**: Always validate features on held-out data

### Common Pitfalls

- **Data Leakage**: Using information from test set in training features
- **Overfitting**: Creating too many features for small datasets
- **Sparsity Issues**: High-dimensional sparse features causing memory problems
- **Scale Mismatch**: Features with different scales affecting model performance
- **Computational Cost**: Expensive features for real-time applications

### Recommended Libraries

```python
# Core feature engineering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA, TruncatedSVD

# Embeddings
from gensim.models import Word2Vec, Doc2Vec
from sentence_transformers import SentenceTransformer

# Advanced NLP
import spacy
from transformers import AutoTokenizer, AutoModel

# Utilities
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
```

## Next Steps

After creating features:

1. **Model Training**: Train models on your engineered features
2. **Hyperparameter Tuning**: Optimize model parameters
3. **Model Evaluation**: Assess performance with appropriate metrics
4. **Feature Importance**: Analyze which features contribute most to predictions
5. **Iteration**: Refine features based on model performance

Continue to [Model Training](./model-training.md) to learn how to train NLP models on your engineered features.
