import nltk
import re
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# === Ensure required NLTK resources ===
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# === Force init: tokenize dummy sentence to warm up punkt ===
_ = sent_tokenize("NLTK init test.")

# === Init tools ===
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
