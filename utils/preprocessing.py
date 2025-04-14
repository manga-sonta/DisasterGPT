import nltk
import re
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# === Ensure required NLTK resources are available (fixes punkt_tab bug) ===
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def wn_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    tokens = word_tokenize(re.sub(r'[-]', ' ', text))
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in [';', '(', ')', '{', '}', ',', '.']]
    pos_tags = pos_tag(filtered_tokens)
    lemmatized = [lemmatizer.lemmatize(word, wn_tagger(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized)
