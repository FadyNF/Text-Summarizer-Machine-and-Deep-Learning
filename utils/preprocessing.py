import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


# ========== TEXT PROCESSING FUNCTIONS ==========

def to_lowercase(text: str) -> str:
    return text.lower()

def remove_control_characters(text: str) -> str:
    return re.sub(r'[\x00-\x1f\x7f-\x9f\u200b]', '', text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_text(text: str) -> list:
    return word_tokenize(text)

def clean_tokens(tokens: list) -> list:
    return [t for t in tokens if t.isalpha() or ('.' in t and len(t) > 1 and not t.strip('.') == '')]

def remove_stopwords(tokens: list) -> list:
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t.lower() not in stop_words]

def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def lemmatize_tokens(tokens: list) -> list:
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(t, get_wordnet_pos(tag)) for t, tag in pos_tags]

def join_tokens(tokens: list) -> str:
    return ' '.join(tokens)

def basic_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = to_lowercase(text)
    text = remove_control_characters(text)
    text = normalize_whitespace(text)
    tokens = tokenize_text(text)
    tokens = clean_tokens(tokens)
    tokens = lemmatize_tokens(tokens)
    tokens = remove_stopwords(tokens)

    return join_tokens(tokens)

# ========== FILE HANDLING ==========

def preprocess_document(input_path: str, output_path: str) -> bool:
    if not os.path.exists(input_path):
        print(f"[Error] File not found at: {os.path.abspath(input_path)}")
        return False

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        clean_text = basic_preprocess(raw_text)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)

        print(f"[Success] Output saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[Error] {e}")
        return False

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    input_path = "../samples/sample2.txt"   
    output_path = "../samples/preprocessed_text2.txt"
    preprocess_document(input_path, output_path)
