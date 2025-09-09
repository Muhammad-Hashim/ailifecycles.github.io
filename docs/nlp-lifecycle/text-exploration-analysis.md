# Text Exploration & Analysis

Once you have collected your text data, the next crucial step is to explore and analyze it thoroughly. This phase helps you understand the characteristics of your dataset, identify potential issues, and gain insights that will inform your preprocessing and modeling decisions.

## Initial Data Inspection

### Basic Statistics

Start by examining fundamental properties of your text data:

```python
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('your_text_data.csv')

# Basic information
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Text column statistics
text_column = 'text'  # Replace with your actual text column name
print(f"Total documents: {len(df)}")
print(f"Missing values: {df[text_column].isnull().sum()}")
print(f"Duplicate texts: {df[text_column].duplicated().sum()}")

# Text length statistics
df['text_length'] = df[text_column].str.len()
print(f"Average text length: {df['text_length'].mean():.1f} characters")
print(f"Median text length: {df['text_length'].median():.1f} characters")
print(f"Min text length: {df['text_length'].min()}")
print(f"Max text length: {df['text_length'].max()}")
```

### Text Length Distribution

Analyze the distribution of text lengths to understand your data's characteristics:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot text length distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['text_length'], bins=50, kde=True)
plt.title('Text Length Distribution')
plt.xlabel('Text Length (characters)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['text_length'])
plt.title('Text Length Box Plot')
plt.ylabel('Text Length (characters)')

plt.tight_layout()
plt.show()

# Percentiles
print("Text length percentiles:")
print(df['text_length'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
```

## Text Quality Analysis

### Language Detection

Ensure your texts are in the expected language(s):

```python
from langdetect import detect, LangDetectError
import langid

def detect_language(text):
    try:
        return detect(text)
    except LangDetectError:
        return 'unknown'

# Apply language detection
df['detected_language'] = df[text_column].apply(detect_language)

# Analyze language distribution
language_counts = df['detected_language'].value_counts()
print("Language distribution:")
print(language_counts)

# Visualize
language_counts.head(10).plot(kind='bar', figsize=(10, 6))
plt.title('Top 10 Languages in Dataset')
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

### Encoding Issues

Check for encoding problems and special characters:

```python
import re

def check_encoding_issues(text):
    # Check for common encoding issues
    issues = []

    # Non-ASCII characters
    if re.search(r'[^\x00-\x7F]', str(text)):
        issues.append('non_ascii')

    # HTML entities
    if re.search(r'&[a-zA-Z]+;', str(text)):
        issues.append('html_entities')

    # Escaped characters
    if '\\' in str(text):
        issues.append('escaped_chars')

    return issues if issues else ['clean']

# Apply encoding check
df['encoding_issues'] = df[text_column].apply(check_encoding_issues)

# Analyze issues
from collections import Counter
all_issues = [issue for issues in df['encoding_issues'] for issue in issues]
issue_counts = Counter(all_issues)
print("Encoding issues found:")
for issue, count in issue_counts.items():
    print(f"{issue}: {count}")
```

## Content Analysis

### Word Frequency Analysis

Analyze the most common words and terms:

```python
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def get_word_frequencies(texts, language='english', max_words=100):
    # Combine all texts
    all_text = ' '.join(str(text) for text in texts if text)

    # Tokenize
    tokens = word_tokenize(all_text.lower())

    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words(language))
    filtered_tokens = [word for word in tokens
                      if word.isalpha() and word not in stop_words and len(word) > 2]

    # Get word frequencies
    word_freq = Counter(filtered_tokens)

    return word_freq.most_common(max_words)

# Get top words
top_words = get_word_frequencies(df[text_column])
print("Top 20 most frequent words:")
for word, freq in top_words[:20]:
    print(f"{word}: {freq}")

# Visualize
words, freqs = zip(*top_words[:20])
plt.figure(figsize=(12, 6))
plt.bar(words, freqs)
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### N-gram Analysis

Analyze common phrases and word combinations:

```python
from nltk.util import ngrams
from collections import Counter

def get_ngrams(texts, n=2, top_k=20):
    all_ngrams = []

    for text in texts:
        if not text:
            continue

        tokens = word_tokenize(str(text).lower())
        # Remove stopwords and non-alphabetic
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens
                          if word.isalpha() and word not in stop_words and len(word) > 2]

        # Generate n-grams
        text_ngrams = list(ngrams(filtered_tokens, n))
        all_ngrams.extend(text_ngrams)

    # Count frequencies
    ngram_freq = Counter(all_ngrams)

    return ngram_freq.most_common(top_k)

# Get bigrams and trigrams
bigrams = get_ngrams(df[text_column], n=2)
trigrams = get_ngrams(df[text_column], n=3)

print("Top 10 bigrams:")
for bigram, freq in bigrams[:10]:
    print(f"{' '.join(bigram)}: {freq}")

print("\nTop 10 trigrams:")
for trigram, freq in trigrams[:10]:
    print(f"{' '.join(trigram)}: {freq}")
```

## Sentiment and Tone Analysis

### Basic Sentiment Distribution

If you have labeled sentiment data, analyze the distribution:

```python
# Assuming you have a 'sentiment' column
if 'sentiment' in df.columns:
    sentiment_counts = df['sentiment'].value_counts()
    print("Sentiment distribution:")
    print(sentiment_counts)

    # Visualize
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
    plt.title('Sentiment Distribution')
    plt.ylabel('')
    plt.show()
```

### Text Complexity Analysis

Analyze readability and complexity metrics:

```python
import textstat

def analyze_text_complexity(text):
    if not text or len(text) < 10:
        return {}

    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'smog_index': textstat.smog_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'difficult_words': textstat.difficult_words(text),
        'linsear_write_formula': textstat.linsear_write_formula(text),
        'gunning_fog': textstat.gunning_fog(text),
        'text_standard': textstat.text_standard(text)
    }

# Apply complexity analysis to a sample
sample_df = df.sample(min(1000, len(df)))  # Analyze sample for speed
complexity_results = sample_df[text_column].apply(analyze_text_complexity)

# Convert to DataFrame
complexity_df = pd.DataFrame(list(complexity_results))
print("Text complexity statistics:")
print(complexity_df.describe())
```

## Data Quality Issues Detection

### Duplicate Detection

Identify and analyze duplicate content:

```python
# Exact duplicates
exact_duplicates = df[df[text_column].duplicated(keep=False)]
print(f"Number of exact duplicate texts: {len(exact_duplicates)}")

# Near-duplicate detection (using similarity)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_texts(texts, threshold=0.9, sample_size=1000):
    # Sample for performance
    sample_texts = texts.sample(min(sample_size, len(texts)))

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sample_texts)

    # Calculate similarities
    similarities = cosine_similarity(tfidf_matrix)

    # Find highly similar pairs
    similar_pairs = []
    for i in range(len(similarities)):
        for j in range(i+1, len(similarities)):
            if similarities[i, j] > threshold:
                similar_pairs.append((i, j, similarities[i, j]))

    return similar_pairs

# Find near-duplicates
similar_pairs = find_similar_texts(df[text_column])
print(f"Number of highly similar text pairs: {len(similar_pairs)}")
```

### Spam and Low-Quality Content Detection

Identify potentially problematic content:

```python
def detect_quality_issues(text):
    issues = []

    if not text:
        return ['empty']

    text_str = str(text)

    # Very short texts
    if len(text_str.split()) < 3:
        issues.append('too_short')

    # Very long texts (potential spam)
    if len(text_str) > 10000:
        issues.append('too_long')

    # High repetition
    words = text_str.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            issues.append('high_repetition')

    # All caps
    if text_str.isupper() and len(text_str) > 10:
        issues.append('all_caps')

    # Excessive punctuation
    punctuation_ratio = sum(1 for char in text_str if char in '!?.,;:-')
    if len(text_str) > 0 and punctuation_ratio / len(text_str) > 0.1:
        issues.append('excessive_punctuation')

    return issues if issues else ['good_quality']

# Apply quality check
df['quality_issues'] = df[text_column].apply(detect_quality_issues)

# Analyze quality issues
quality_counts = Counter([issue for issues in df['quality_issues'] for issue in issues])
print("Quality issues detected:")
for issue, count in quality_counts.items():
    print(f"{issue}: {count}")
```

## Domain-Specific Analysis

### Topic Modeling

Discover latent topics in your text data:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def perform_topic_modeling(texts, n_topics=5, n_words=10):
    # Preprocess texts
    processed_texts = []
    for text in texts:
        if text:
            tokens = word_tokenize(str(text).lower())
            stop_words = set(stopwords.words('english'))
            filtered = [word for word in tokens
                       if word.isalpha() and word not in stop_words and len(word) > 2]
            processed_texts.append(' '.join(filtered))

    # Vectorize
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    dtm = vectorizer.fit_transform(processed_texts)

    # Fit LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words-1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    return topics

# Perform topic modeling on a sample
sample_texts = df[text_column].dropna().sample(min(5000, len(df)))
topics = perform_topic_modeling(sample_texts)

print("Discovered topics:")
for topic in topics:
    print(topic)
```

## Visualization and Reporting

### Create Comprehensive Analysis Report

```python
def generate_analysis_report(df, text_column):
    report = {
        'basic_stats': {
            'total_documents': len(df),
            'missing_texts': df[text_column].isnull().sum(),
            'duplicate_texts': df[text_column].duplicated().sum(),
            'avg_text_length': df[text_column].str.len().mean(),
            'median_text_length': df[text_column].str.len().median()
        },
        'quality_metrics': {
            'empty_texts': (df[text_column].str.len() == 0).sum(),
            'very_short_texts': (df[text_column].str.split().str.len() < 3).sum(),
            'very_long_texts': (df[text_column].str.len() > 10000).sum()
        }
    }

    return report

# Generate report
analysis_report = generate_analysis_report(df, text_column)
print("Analysis Report:")
for category, metrics in analysis_report.items():
    print(f"\n{category.upper()}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
```

## Best Practices for Text Exploration

### Systematic Approach

1. **Start with Basic Statistics**: Get a high-level understanding
2. **Check Data Quality**: Identify and document issues early
3. **Analyze Content**: Understand topics, sentiment, and patterns
4. **Consider Domain Context**: Apply domain-specific analysis
5. **Document Findings**: Keep detailed records for reproducibility

### Common Pitfalls to Avoid

- **Not checking for duplicates**: Can skew analysis and model training
- **Ignoring text length variations**: May affect preprocessing decisions
- **Overlooking encoding issues**: Can cause processing errors
- **Missing quality checks**: Poor quality data leads to poor models
- **Not sampling appropriately**: Large datasets need careful sampling

### Tools and Libraries

```python
# Core NLP libraries
import nltk
import spacy

# Text analysis
import textstat
from langdetect import detect
import langid

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Topic modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Statistical analysis
import pandas as pd
import numpy as np
from collections import Counter
```

## Next Steps

After thorough exploration and analysis:

1. **Data Cleaning**: Address identified quality issues
2. **Preprocessing Pipeline**: Design based on your findings
3. **Feature Engineering**: Plan text representation strategies
4. **Baseline Models**: Establish performance benchmarks

Continue to [Text Preprocessing & Preparation](./text-preprocessing-preparation.md) to begin cleaning and preparing your data for modeling.

### 1. Language Detection

```python
from langdetect import detect, LangDetectError

def detect_language(text):
    try:
        return detect(text)
    except LangDetectError:
        return 'unknown'

# Detect languages in your corpus
df['language'] = df['text'].apply(detect_language)
language_counts = df['language'].value_counts()
print("Language distribution:")
print(language_counts)
```

### 2. Most Common Words

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('stopwords')

def preprocess_for_frequency(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return words

# Get all words from the corpus
all_words = []
for text in df['text']:
    all_words.extend(preprocess_for_frequency(text))

# Most common words
word_freq = Counter(all_words)
most_common = word_freq.most_common(20)

# Visualize
words, freqs = zip(*most_common)
plt.figure(figsize=(12, 6))
plt.bar(words, freqs)
plt.title('Top 20 Most Common Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 3. Word Clouds

```python
from wordcloud import WordCloud

# Create word cloud
text_for_cloud = ' '.join(df['text'])
wordcloud = WordCloud(width=800, height=400, 
                     background_color='white',
                     max_words=100,
                     colormap='viridis').generate(text_for_cloud)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Text Data')
plt.show()
```

## Label Distribution Analysis

### For Classification Tasks

```python
# Analyze label distribution
if 'label' in df.columns:
    label_counts = df['label'].value_counts()
    
    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.index, label_counts.values)
    plt.title('Distribution of Labels')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Check for class imbalance
    print("Class distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")
```

## Text Quality Assessment

### 1. Identify Potential Issues

```python
# Find very short texts
short_texts = df[df['text'].str.len() < 10]
print(f"Very short texts (< 10 chars): {len(short_texts)}")

# Find very long texts
long_texts = df[df['text'].str.len() > 5000]
print(f"Very long texts (> 5000 chars): {len(long_texts)}")

# Find texts with mostly numbers or special characters
def is_low_quality(text):
    if pd.isna(text):
        return True
    # Check if mostly non-alphabetic
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    return alpha_ratio < 0.5

low_quality = df[df['text'].apply(is_low_quality)]
print(f"Low quality texts: {len(low_quality)}")
```

### 2. Duplicate Detection

```python
# Find exact duplicates
duplicates = df[df.duplicated('text', keep=False)]
print(f"Exact duplicates: {len(duplicates)}")

# Find near-duplicates (simplified approach)
from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Sample approach for small datasets
# For large datasets, use more efficient methods like MinHash
```

## Advanced Text Analysis

### 1. Sentiment Distribution

```python
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Calculate sentiment for each text
df['sentiment'] = df['text'].apply(get_sentiment)

# Plot sentiment distribution
plt.figure(figsize=(10, 6))
plt.hist(df['sentiment'], bins=30, edgecolor='black')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score (-1: Negative, 1: Positive)')
plt.ylabel('Frequency')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.show()
```

### 2. Topic Modeling (Quick Overview)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Simple topic modeling
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text'].fillna(''))

# LDA for topic discovery
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# Display top words for each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:]]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

## Visualization Dashboard

### Creating Interactive Plots

```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive text length distribution
fig = px.histogram(df, x=df['text'].str.len(), 
                   title='Interactive Text Length Distribution')
fig.show()

# If you have labels, create interactive label distribution
if 'label' in df.columns:
    fig = px.bar(df['label'].value_counts(), 
                 title='Label Distribution')
    fig.show()
```

## Key Insights to Look For

### Data Quality Issues

- Very short or very long texts
- Duplicate or near-duplicate content
- Missing or corrupted data
- Language inconsistencies

### Content Characteristics

- Vocabulary diversity and complexity
- Domain-specific terminology
- Informal vs formal language
- Temporal patterns (if timestamps available)

### Class Balance (for supervised tasks)

- Severe class imbalance requiring resampling
- Sufficient examples per class
- Representative distribution
