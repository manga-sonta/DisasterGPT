import nltk
import re
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# No word_tokenize used now

# === Ensure required NLTK resources ===
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

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
    # Tokenize using regex instead of word_tokenize (no punkt needed!)
    tokens = re.findall(r'\b\w+\b', re.sub(r'[-]', ' ', text))

    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    pos_tags = pos_tag(filtered_tokens)
    lemmatized = [lemmatizer.lemmatize(word, wn_tagger(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized)
