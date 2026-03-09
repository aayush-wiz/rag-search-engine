#!/usr/bin/env python3
"""
Keyword Search CLI: A tool for indexing and searching movie data using BM25 and TF-IDF.

This script provides a command-line interface to build an inverted index from a movie dataset
and perform various operations like keyword search, term frequency (TF), inverse document
frequency (IDF), and BM25 score calculations.
"""

import argparse
import os
import sys

# Add the project root to sys.path to allow importing from the 'cli', 'lib', and 'utils' directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cli.utils.paths import DATA_PATH, STOP_WORDS_PATH, CACHE_DIR  # noqa: E402
from cli.lib.keyword_search import (  # noqa: E402
    InvertedIndex,
    bm25_idf_command,
    bm25_tf_command,
    load_resources,
)
from cli.utils.search_utils import BM25_B, BM25_K1, clean_and_tokenize  # noqa: E402


STOP_WORDS, MOVIES = load_resources()


def main() -> None:
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Movie Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- CLI Command Definitions ---

    # 'build' command: Creates the index from scratch
    subparsers.add_parser("build", help="Index the movie dataset and save to cache")

    # 'search' command: Basic keyword search
    search_parser = subparsers.add_parser("search", help="Search for movies by keyword")
    search_parser.add_argument(
        "query", type=str, help="Search query (phrase or single word)"
    )

    # 'tf' command: Get raw term frequency
    tf_parser = subparsers.add_parser("tf", help="Get raw term frequency for a movie")
    tf_parser.add_argument("doc_id", type=int, help="Movie ID")
    tf_parser.add_argument("term", type=str, help="Term to check frequency for")

    # 'idf' command: Standard IDF
    idf_parser = subparsers.add_parser("idf", help="Get IDF score for a term")
    idf_parser.add_argument("term", type=str, help="Prints the idf value for this term")

    # 'tfidf' command: Combine TF and IDF
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score of a term in a movie"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Movie ID")
    tfidf_parser.add_argument("term", type=str, help="The term to check frequency for")

    # 'bm25idf' command: BM25 variant of IDF
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    # 'bm25tf' command: BM25 variant of TF
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a movie/term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable k1 param"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit",
        type=int,
        default=5,  # Set a default value if the argument is not provided
        help="The maximum number of movies to process (default: 10).",
    )

    args = parser.parse_args()

    index = InvertedIndex()

    match args.command:
        case "search":
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            query_tokens = clean_and_tokenize(args.query, STOP_WORDS)
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

        case "tf":
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            try:
                freq = index.get_tf(args.doc_id, args.term, STOP_WORDS)
                movie_title = index.docmap.get(args.doc_id, {}).get(
                    "title", f"Movie {args.doc_id}"
                )
                print(f"Term '{args.term}' appears {freq} time(s) in '{movie_title}'")
            except ValueError as e:
                print(f"Error: {e}")

        case "idf":
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Tip: Run './cli/keyword_search_cli.py build' first.")
                return

            try:
                idf = index.get_idf(args.term, STOP_WORDS)
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            except ValueError as e:
                print(f"Error: {e}")

        case "tfidf":
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            try:
                tf_idf = index.get_tfidf(args.doc_id, args.term, STOP_WORDS)
                print(
                    f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
                )
            except ValueError as e:
                print(f"Error: {e}")

        case "bm25idf":
            result = bm25_idf_command(args.term, STOP_WORDS)
            if result is not None:
                print(f"BM25 IDF score of '{args.term}': {result:.2f}")

        case "bm25tf":
            result = bm25_tf_command(
                args.doc_id, args.term, STOP_WORDS, args.k1, args.b
            )
            if result is not None:
                print(
                    f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {result:.2f}"
                )

        case "bm25search":
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Tip: Run './cli/keyword_search_cli.py build' first.")
                return

            try:
                results = index.bm25_search(args.query, STOP_WORDS, args.limit)
                if not results:
                    print("No results found.")
                    return

                for i, (doc_id, score) in enumerate(results, 1):
                    movie = index.docmap.get(doc_id, {"title": "Unknown", "id": doc_id})
                    print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")
            except ValueError as e:
                print(f"Error: {e}")

        case "build":
            index.build(MOVIES, STOP_WORDS)
            index.save()
            print("Index built and saved.")


if __name__ == "__main__":
    main()
