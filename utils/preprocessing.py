import re
import spacy
from nltk.corpus import stopwords

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# === Setup ===
# Use spaCy's small English model (make sure to install it!)
nlp = spacy.load("en_core_web_sm")

# Just for stopword removal
import nltk
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# === Preprocessing ===
def preprocess_text(text):
    # Tokenize using regex first to clean input
    tokens = re.findall(r'\b\w+\b', re.sub(r'[-]', ' ', text))
    filtered = [word.lower() for word in tokens if word.lower() not in stop_words]
    
    # Feed filtered text into spaCy
    doc = nlp(" ".join(filtered))

    # Lemmatize using POS from spaCy
    lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return ' '.join(lemmatized)
