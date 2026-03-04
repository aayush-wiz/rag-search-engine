#!/usr/bin/env python3

import argparse
import json
import string
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from collections import Counter
from nltk.stem import PorterStemmer
import pickle
import math

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "movies.json"
STOP_WORDS_PATH = ROOT / "data" / "stopwords.txt"
CACHE_DIR = ROOT / "cache"
INDEX_CACHE_PATH = CACHE_DIR / "index.pkl"
DOCMAP_CACHE_PATH = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_CACHE_PATH = CACHE_DIR / "term_frequencies.pkl"

STEMMER = PorterStemmer()
TRANSLATOR = str.maketrans("", "", string.punctuation)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: Dict[str, Set[int]] = {}  # string -> set(int)
        self.docmap: Dict[int, Dict[str, Any]] = {}  # int -> dict(string)
        self.term_frequencies: Dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        text_tokens = clean_and_tokenize(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in text_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> List[int]:
        clean_term = term.lower()
        return sorted(list(self.index.get(clean_term, set())))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = clean_and_tokenize(term)
        if not tokens:
            return 0
        if len(tokens) > 1:
            raise ValueError("The 'tf' command only supports single-term queries.")

        token = tokens[0]
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(token, 0)

    def build(self) -> None:
        for movie in MOVIES:
            doc_id = movie["id"]
            combined_text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, combined_text)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(INDEX_CACHE_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_CACHE_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_FREQUENCIES_CACHE_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        if not INDEX_CACHE_PATH.exists() or not DOCMAP_CACHE_PATH.exists():
            raise FileNotFoundError(
                "Cache files do not exist. Please run 'build' first."
            )
        with open(INDEX_CACHE_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_CACHE_PATH, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FREQUENCIES_CACHE_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)


def load_resources() -> Tuple[Set[str], List[Dict[str, Any]]]:
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


STOP_WORDS, MOVIES = load_resources()


def clean_and_tokenize(text: str) -> List[str]:
    """Clean, tokenize, and stem text efficiently."""
    tokens = text.lower().translate(TRANSLATOR).split()
    return [STEMMER.stem(w) for w in tokens if w not in STOP_WORDS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build command
    subparsers.add_parser("build", help="Build the inverted index")

    # search command
    search_parser = subparsers.add_parser(
        "search", help="Search for movies matching query"
    )
    search_parser.add_argument("query", type=str, help="Search query")

    # tf command
    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a specific movie"
    )
    tf_parser.add_argument("doc_id", type=int, help="The ID of the movie")
    tf_parser.add_argument("term", type=str, help="The term to check frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get the inverted Document frequency"
    )
    idf_parser.add_argument("term", type=str, help="Prints the idf value for this term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get the highest TF-IDF value of term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="The ID of the movie")
    tfidf_parser.add_argument("term", type=str, help="The term to check frequency for")

    args = parser.parse_args()

    index = InvertedIndex()

    if args.command == "search":
        try:
            index.load()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        query_tokens = clean_and_tokenize(args.query)
        results = []
        seen_ids = set()

        for token in query_tokens:
            doc_ids = index.get_documents(token)
            for doc_id in doc_ids:
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    results.append(doc_id)
        if not results:
            print("No results found.")
            return
        for i, doc_id in enumerate(results, 1):
            movie = index.docmap[doc_id]
            print(f"{i}. {movie['title']} (ID: {doc_id})")

    elif args.command == "tf":
        try:
            index.load()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        try:
            freq = index.get_tf(args.doc_id, args.term)
            movie_title = index.docmap.get(args.doc_id, {}).get(
                "title", f"Movie {args.doc_id}"
            )
            print(f"Term '{args.term}' appears {freq} time(s) in '{movie_title}'")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.command == "idf":
        try:
            index.load()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Tip: Run './cli/keyword_search_cli.py build' first.")
            return

        try:
            total_doc_count = len(index.docmap)
            if total_doc_count == 0:
                print("Error: The index is empty. Please run 'build' first.")
                return

            tokens = clean_and_tokenize(args.term)
            if not tokens:
                print(f"Inverse document frequency of '{args.term}': 0.00 (Stopword)")
                return

            term = tokens[0]
            matching_docs_count = len(index.get_documents(term))

            # Standard smooth IDF formula
            idf = math.log((total_doc_count + 1) / (matching_docs_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.command == "tfidf":
        try:
            index.load()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        try:
            tf = index.get_tf(args.doc_id, args.term)

            # Calculation for IDF
            total_doc_count = len(index.docmap)
            tokens = clean_and_tokenize(args.term)
            if not tokens or total_doc_count == 0:
                idf = 0.0
            else:
                term = tokens[0]
                matching_docs_count = len(index.get_documents(term))
                idf = math.log((total_doc_count + 1) / (matching_docs_count + 1))

            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        except ValueError as e:
            print(f"Error: {e}")

    elif args.command == "build":
        index.build()
        index.save()
        print("Index built and saved.")


if __name__ == "__main__":
    main()
