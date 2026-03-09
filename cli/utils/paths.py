"""Centralized path definitions for the RAG Search Engine project."""

import os

# Get the project root directory (parent of cli/)
CLI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(CLI_DIR)

# Data directories
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "cache")

# Data file paths
DATA_PATH = os.path.join(DATA_DIR, "movies.json")
STOP_WORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")

# Cache file paths
INDEX_CACHE_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_CACHE_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_CACHE_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTH_CACHE_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")
EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
