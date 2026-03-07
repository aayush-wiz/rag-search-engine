import string
from typing import List, Set
from nltk.stem import PorterStemmer

BM25_B = 0.75
BM25_K1 = 1.5

STEMMER = PorterStemmer()
TRANSLATOR = str.maketrans("", "", string.punctuation)


def clean_and_tokenize(text: str, stop_words: Set[str]) -> List[str]:
    """Clean, tokenize, and stem text efficiently."""
    tokens = text.lower().translate(TRANSLATOR).split()
    return [STEMMER.stem(w) for w in tokens if w not in stop_words]
