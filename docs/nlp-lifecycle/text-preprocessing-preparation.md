# Text Preprocessing & Preparation

Text preprocessing is a critical step in NLP that transforms raw text into a format suitable for machine learning models. This phase involves cleaning, normalizing, and preparing text data to improve model performance and reduce noise.

## Text Cleaning

### Basic Text Cleaning

Remove unwanted characters, formatting, and noise from your text:

```python
import re
import string

def basic_text_cleaning(text):
    """
    Perform basic text cleaning operations
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

# Apply to your dataset
df['cleaned_text'] = df['text'].apply(basic_text_cleaning)
```

### Advanced Cleaning Techniques

Handle more complex text issues:

```python
def advanced_text_cleaning(text):
    """
    Advanced text cleaning with multiple techniques
    """
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"  # dingbats
        "\U0001f926-\U0001f937"  # gestures
        "\U00010000-\U0010ffff"  # other unicode
        "\u2640-\u2642"  # gender symbols
        "\u2600-\u2B55"  # misc symbols
        "\u200d"  # zero width joiner
        "\u23cf"  # eject symbol
        "\u23e9"  # fast forward
        "\u231a"  # watch
        "\ufe0f"  # variation selector
        "\u3030"  # wavy dash
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Remove non-ASCII characters (optional, be careful with multilingual data)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove numbers (optional)
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text
```

## Tokenization

### Word Tokenization

Split text into individual words or tokens:

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
nltk.download('punkt')

def tokenize_words(text):
    """
    Tokenize text into words
    """
    return word_tokenize(text)

def tokenize_sentences(text):
    """
    Tokenize text into sentences
    """
    return sent_tokenize(text)

# Apply tokenization
df['word_tokens'] = df['cleaned_text'].apply(tokenize_words)
df['sentence_tokens'] = df['text'].apply(tokenize_sentences)

# Example output
sample_text = "Hello world! This is a sample sentence."
print("Word tokens:", tokenize_words(sample_text))
print("Sentence tokens:", tokenize_sentences(sample_text))
```

### Subword Tokenization

Use advanced tokenization for modern NLP models:

```python
from transformers import AutoTokenizer

def subword_tokenization(text, model_name='bert-base-uncased'):
    """
    Perform subword tokenization using transformers
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=True)

    return {
        'tokens': tokens,
        'token_ids': token_ids,
        'decoded': tokenizer.decode(token_ids)
    }

# Example
text = "Hello world! This is tokenization."
result = subword_tokenization(text)
print("Subword tokens:", result['tokens'])
print("Token IDs:", result['token_ids'])
```

## Normalization

### Stemming

Reduce words to their root form:

```python
from nltk.stem import PorterStemmer, SnowballStemmer

def stem_text(text, stemmer_type='porter'):
    """
    Apply stemming to text
    """
    if stemmer_type == 'porter':
        stemmer = PorterStemmer()
    elif stemmer_type == 'snowball':
        stemmer = SnowballStemmer('english')
    else:
        raise ValueError("Invalid stemmer type")

    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed_tokens)

# Example
text = "running runs runner"
print("Original:", text)
print("Stemmed:", stem_text(text))
```

### Lemmatization

Reduce words to their base dictionary form:

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    """
    Apply lemmatization with POS tagging
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)

    lemmatized_tokens = []
    for token in tokens:
        pos = get_wordnet_pos(token)
        lemma = lemmatizer.lemmatize(token, pos)
        lemmatized_tokens.append(lemma)

    return ' '.join(lemmatized_tokens)

# Example
text = "running better good"
print("Original:", text)
print("Lemmatized:", lemmatize_text(text))
```

## Stopword Removal

Remove common words that don't carry much meaning:

```python
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download NLTK stopwords
nltk.download('stopwords')

def remove_stopwords(text, stopword_source='nltk'):
    """
    Remove stopwords from text
    """
    if stopword_source == 'nltk':
        stop_words = set(stopwords.words('english'))
    elif stopword_source == 'sklearn':
        stop_words = ENGLISH_STOP_WORDS
    else:
        raise ValueError("Invalid stopword source")

    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    return ' '.join(filtered_tokens)

# Example
text = "This is a sample sentence with some common words"
print("Original:", text)
print("Without stopwords:", remove_stopwords(text))

# Custom stopwords
custom_stopwords = {'sample', 'common'}
def remove_custom_stopwords(text, custom_stops):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in custom_stops]
    return ' '.join(filtered_tokens)
```

## Text Normalization Techniques

### Case Normalization

```python
def normalize_case(text, method='lower'):
    """
    Normalize text case
    """
    if method == 'lower':
        return text.lower()
    elif method == 'upper':
        return text.upper()
    elif method == 'title':
        return text.title()
    else:
        return text
```

### Unicode Normalization

```python
import unicodedata

def unicode_normalize(text, form='NFKC'):
    """
    Normalize unicode characters
    """
    return unicodedata.normalize(form, text)

# Example
text = "café"
print("Original:", repr(text))
print("Normalized:", repr(unicode_normalize(text)))
```

### Number and Symbol Handling

```python
def handle_numbers_symbols(text, number_handling='keep', symbol_handling='keep'):
    """
    Handle numbers and symbols in text
    """
    if number_handling == 'remove':
        text = re.sub(r'\d+', '', text)
    elif number_handling == 'replace':
        text = re.sub(r'\d+', '<NUM>', text)

    if symbol_handling == 'remove':
        text = re.sub(r'[^\w\s]', '', text)
    elif symbol_handling == 'replace':
        text = re.sub(r'[^\w\s]', '<SYM>', text)

    return text
```

## Language-Specific Processing

### Multilingual Text Handling

```python
from langdetect import detect
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.fr import French

def language_specific_processing(text, lang=None):
    """
    Apply language-specific preprocessing
    """
    if not lang:
        try:
            lang = detect(text)
        except:
            lang = 'en'  # default to English

    if lang == 'en':
        nlp = English()
    elif lang == 'es':
        nlp = Spanish()
    elif lang == 'fr':
        nlp = French()
    else:
        # Fallback to basic processing
        return basic_text_cleaning(text)

    doc = nlp(text)

    # Language-specific tokenization
    tokens = [token.text for token in doc]

    return ' '.join(tokens)
```

## Preprocessing Pipeline

### Complete Preprocessing Function

```python
def complete_preprocessing_pipeline(text, config=None):
    """
    Complete text preprocessing pipeline
    """
    if config is None:
        config = {
            'lowercase': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_html': True,
            'remove_emojis': True,
            'remove_numbers': False,
            'remove_punctuation': False,
            'remove_stopwords': True,
            'stemming': False,
            'lemmatization': True,
            'min_word_length': 2,
            'max_word_length': 20
        }

    # Basic cleaning
    if config['lowercase']:
        text = text.lower()

    if config['remove_urls']:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    if config['remove_emails']:
        text = re.sub(r'\S+@\S+', '', text)

    if config['remove_html']:
        text = re.sub(r'<.*?>', '', text)

    if config['remove_emojis']:
        emoji_pattern = re.compile("["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove short/long words
    tokens = [token for token in tokens
              if config['min_word_length'] <= len(token) <= config['max_word_length']]

    # Remove numbers
    if config['remove_numbers']:
        tokens = [token for token in tokens if not token.isdigit()]

    # Remove punctuation
    if config['remove_punctuation']:
        tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    if config['remove_stopwords']:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]

    # Stemming or Lemmatization
    if config['stemming']:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    elif config['lemmatization']:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back
    processed_text = ' '.join(tokens)

    return processed_text

# Apply to dataset
preprocessing_config = {
    'lowercase': True,
    'remove_urls': True,
    'remove_emails': True,
    'remove_html': True,
    'remove_emojis': True,
    'remove_numbers': False,
    'remove_punctuation': False,
    'remove_stopwords': True,
    'stemming': False,
    'lemmatization': True,
    'min_word_length': 2,
    'max_word_length': 20
}

df['processed_text'] = df['cleaned_text'].apply(
    lambda x: complete_preprocessing_pipeline(x, preprocessing_config)
)
```

## Quality Control and Validation

### Preprocessing Validation

```python
def validate_preprocessing(original_texts, processed_texts):
    """
    Validate preprocessing results
    """
    validation_results = {
        'total_texts': len(original_texts),
        'empty_after_processing': sum(1 for text in processed_texts if not text.strip()),
        'avg_original_length': sum(len(text) for text in original_texts) / len(original_texts),
        'avg_processed_length': sum(len(text) for text in processed_texts) / len(processed_texts),
        'vocab_size_original': len(set(' '.join(original_texts).split())),
        'vocab_size_processed': len(set(' '.join(processed_texts).split()))
    }

    return validation_results

# Validate preprocessing
validation = validate_preprocessing(df['cleaned_text'].tolist(), df['processed_text'].tolist())
print("Preprocessing Validation:")
for metric, value in validation.items():
    print(f"{metric}: {value}")
```

## Best Practices

### Preprocessing Guidelines

1. **Start Simple**: Begin with basic cleaning and gradually add complexity
2. **Domain Awareness**: Consider domain-specific requirements and terminology
3. **Language Considerations**: Handle multilingual data appropriately
4. **Performance Trade-offs**: Balance preprocessing quality with computational cost
5. **Reproducibility**: Document all preprocessing steps and parameters

### Common Pitfalls

- **Over-preprocessing**: Removing too much information
- **Under-preprocessing**: Leaving too much noise
- **Inconsistent Processing**: Different preprocessing for train/test sets
- **Language Bias**: Assuming English-only processing
- **Performance Issues**: Inefficient preprocessing for large datasets

### Recommended Libraries

```python
# Core NLP libraries
import nltk
import spacy
from textblob import TextBlob

# Advanced preprocessing
from transformers import AutoTokenizer
from cleantext import clean
import ftfy  # Fix text encoding issues

# Parallel processing
from multiprocessing import Pool
import dask.dataframe as dd
```

## Next Steps

After preprocessing your text data:

1. **Feature Engineering**: Convert text to numerical features
2. **Model Selection**: Choose appropriate NLP models
3. **Training**: Train your models on preprocessed data
4. **Evaluation**: Assess model performance
5. **Iteration**: Refine preprocessing based on model results

Continue to [Feature Engineering](./feature-engineering.md) to learn how to convert your preprocessed text into numerical features for machine learning models.
