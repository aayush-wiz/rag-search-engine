"""
Keyword Search Library: Core logic for indexing and searching movie data.
"""

import json
import math
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.search_utils import BM25_B, BM25_K1, clean_and_tokenize

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(LIB_DIR)

DATA_PATH = os.path.join(ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(ROOT, "cache")
INDEX_CACHE_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_CACHE_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_CACHE_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTH_CACHE_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")


class InvertedIndex:
    """
    Core class for managing the search index and performing IR calculations.

    Attributes:
        index (Dict[str, Set[int]]): Maps tokens to sets of document IDs.
        docmap (Dict[int, Dict[str, Any]]): Maps document IDs to movie metadata.
        term_frequencies (Dict[int, Counter]): Maps document IDs to term counts.
        doc_lengths (Dict[int, int]): Maps document IDs to the number of tokens they contain.
    """

    def __init__(self) -> None:
        """Initialize empty data structures for the index."""
        self.index: Dict[str, Set[int]] = {}  # token -> set of doc_ids
        self.docmap: Dict[int, Dict[str, Any]] = {}  # doc_id -> movie metadata
        self.term_frequencies: Dict[
            int, Counter
        ] = {}  # doc_id -> Counter(token: count)
        self.doc_lengths: Dict[int, int] = {}  # doc_id -> token count

    def add_document(self, doc_id: int, text: str, stop_words: Set[str]) -> None:
        """
        Tokenizes text and updates internal structures with the new document.

        Args:
            doc_id: Unique identifier for the movie.
            text: Combined title and description to index.
            stop_words: Set of words to ignore.
        """
        text_tokens = clean_and_tokenize(text, stop_words)

        # Track document length
        self.doc_lengths[doc_id] = len(text_tokens)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in text_tokens:
            # Update inverted index (mapping tokens to documents)
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            # Update term frequencies (counts within the specific document)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> List[int]:
        """
        Retrieves IDs of all documents containing the given term.

        Args:
            term: The keyword to search for.

        Returns:
            A sorted list of document IDs.
        """
        clean_term = term.lower()
        return sorted(list(self.index.get(clean_term, set())))

    def get_tf(self, doc_id: int, term: str, stop_words: Set[str]) -> int:
        """
        Calculates the raw term frequency (TF) of a term in a document.

        Args:
            doc_id: The document to check.
            term: The term to count.
            stop_words: Set of words to ignore.

        Returns:
            The number of times the term appears in the document.
        """
        tokens = clean_and_tokenize(term, stop_words)
        if not tokens:
            return 0
        if len(tokens) > 1:
            raise ValueError("The 'tf' command only supports single-term queries.")

        token = tokens[0]
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(token, 0)

    def get_idf(self, term: str, stop_words: Set[str]) -> float:
        """
        Calculates the standard smooth Inverse Document Frequency (IDF).

        Formula: log((total_docs + 1) / (matching_docs + 1))

        Args:
            term: The term to calculate IDF for.
            stop_words: Set of words to ignore.

        Returns:
            The standard IDF score.
        """
        tokens = clean_and_tokenize(term, stop_words)
        if not tokens:
            return 0.0

        term = tokens[0]
        total_docs = len(self.docmap)
        matching_docs = len(self.index.get(term, set()))

        return math.log((total_docs + 1) / (matching_docs + 1))

    def get_tfidf(self, doc_id: int, term: str, stop_words: Set[str]) -> float:
        """
        Calculates the TF-IDF score for a term in a document.

        Args:
            doc_id: The document ID.
            term: The term.
            stop_words: Set of words to ignore.

        Returns:
            The TF-IDF score.
        """
        tf = self.get_tf(doc_id, term, stop_words)
        idf = self.get_idf(term, stop_words)
        return tf * idf

    def get_bm25_idf(self, term: str, stop_words: Set[str]) -> float:
        """
        Calculates the BM25 variant of Inverse Document Frequency (IDF).

        Args:
            term: The term to calculate IDF for.
            stop_words: Set of words to ignore.

        Returns:
            The BM25 IDF score.
        """
        tokens = clean_and_tokenize(term, stop_words)
        if not tokens:
            return 0.0
        if len(tokens) > 1:
            raise ValueError("Token length should be equal to 1.")

        token = tokens[0]
        n_total = len(self.docmap)
        df = len(self.index.get(token, set()))

        # BM25 IDF formula: log((N - DF + 0.5) / (DF + 0.5) + 1)
        return math.log((n_total - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self,
        doc_id: int,
        term: str,
        stop_words: Set[str],
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> float:
        """
        Calculates the BM25 term frequency score for a document.

        Args:
            doc_id: The document ID.
            term: The term.
            stop_words: Set of words to ignore.
            k1: The tunable parameter (default from search_utils).
            b: The tunable parameter that controls how much we care about document length.

        Returns:
            The BM25 TF contribution.
        """
        tf = self.get_tf(doc_id, term, stop_words)
        avg_dl = self._get_avg_doc_length()

        # If no documents or empty corpus, length normalization can't be calculated
        if avg_dl == 0:
            length_norm = 1.0
        else:
            doc_len = self.doc_lengths.get(doc_id, 0)
            length_norm = (1 - b) + b * (doc_len / avg_dl)

        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return tf_component

    def bm25(self, doc_id: int, term: str, stop_words: Set[str]) -> float:
        """Calculates the BM25 score for a single term in a document."""
        tf = self.get_bm25_tf(doc_id, term, stop_words)
        idf = self.get_bm25_idf(term, stop_words)
        return tf * idf

    def _get_avg_doc_length(self) -> float:
        """
        Calculates the average document length across the entire indexed corpus.

        Returns:
            The average number of tokens per document as a float.
            Returns 0.0 if the index contains no documents.
        """
        if not self.docmap:
            return 0.0

        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.docmap)

    def bm25_search(
        self, query: str, stop_words: Set[str], limit: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Performs a full BM25 search across the indexed corpus.

        Args:
            query: The search query string.
            stop_words: Set of words to ignore.
            limit: Maximum number of results to return.

        Returns:
            A list of (doc_id, score) tuples, sorted by score descending.
        """
        tokens = clean_and_tokenize(query, stop_words)
        scores: Dict[int, float] = {}

        for token in tokens:
            if token not in self.index:
                continue
            for doc_id in self.index[token]:
                score = self.bm25(doc_id, token, stop_words)
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

    def build(self, movies: List[Dict[str, Any]], stop_words: Set[str]) -> None:
        """
        Populates the index from a list of movie dictionaries.

        Args:
            movies: List of movie metadata dictionaries.
            stop_words: Set of words to ignore.
        """
        for movie in movies:
            doc_id = movie["id"]
            # Combine title and description for a richer search space
            combined_text = f"{movie['title']} {movie['description']}"
            self.add_document(doc_id, combined_text, stop_words)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        """Persists the index data to disk in the cache directory."""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        with open(INDEX_CACHE_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_CACHE_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_FREQUENCIES_CACHE_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTH_CACHE_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        """Loads index data from the persistent cache files."""
        if not os.path.exists(INDEX_CACHE_PATH) or not os.path.exists(
            DOCMAP_CACHE_PATH
        ):
            raise FileNotFoundError(
                "Cache files do not exist. Please run 'build' first."
            )
        with open(INDEX_CACHE_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_CACHE_PATH, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FREQUENCIES_CACHE_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)
        if os.path.exists(DOC_LENGTH_CACHE_PATH):
            with open(DOC_LENGTH_CACHE_PATH, "rb") as f:
                self.doc_lengths = pickle.load(f)


def load_resources() -> Tuple[Set[str], List[Dict[str, Any]]]:
    """
    Loads external data files (stop words and movie data).

    Returns:
        A tuple containing (set of stop words, list of movie dictionaries).
    """
    try:
        with open(STOP_WORDS_PATH, "r") as f:
            stop_words = set(f.read().split())
    except FileNotFoundError:
        stop_words = set()

    try:
        with open(DATA_PATH, "r") as f:
            movies = json.load(f).get("movies", [])
    except (FileNotFoundError, json.JSONDecodeError):
        movies = []

    return stop_words, movies


STOP_WORDS, _ = load_resources()


def bm25_idf_command(term: str, stop_words: Set[str]) -> Optional[float]:
    """
    Wrapper for calculating BM25 IDF.
    """
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    return index.get_bm25_idf(term, stop_words)


def bm25_tf_command(
    doc_id: int,
    term: str,
    stop_words: Set[str],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> Optional[float]:
    """
    Wrapper for calculating BM25 TF.
    """
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    return index.get_bm25_tf(doc_id, term, stop_words, k1, b)
