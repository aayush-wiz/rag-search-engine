#!/usr/bin/env python3
"""CLI for semantic search operations including embedding, verification, and chunking."""
import os
import sys

# Add the project root to sys.path to allow importing from the 'cli' and 'lib' directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cli.utils.paths import CACHE_DIR, DATA_PATH  # noqa: E402
from cli.lib.semantic_search import (  # noqa: E402
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    perform_semantic_search,
    chunk_query,
)
import argparse  # noqa: E402


def main() -> None:
    """Entry point for the Semantic Search CLI."""
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "verify", help="Verify the model that is used for text embedding"
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generates text embedding of the given query"
    )
    embed_text_parser.add_argument(
        "query", type=str, help="Query for generating text embedding"
    )

    subparsers.add_parser("verify_embeddings", help="Verifies the text embeddings")

    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Creates vector from the given query"
    )
    embedquery_parser.add_argument(
        "query", type=str, help="Query for generating vector"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search for the top (limit) queries from the document"
    )
    search_parser.add_argument("query", type=str, help="Description for the movie")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="The maximum number of movies to process (default: 5).",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split a query into fixed-size word chunks with optional overlap"
    )
    chunk_parser.add_argument("query", type=str, help="Text to split into chunks")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=5, help="Number of words per chunk (default: 5)"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, help="Number of words to overlap between chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split a query into sentence-based chunks with optional overlap",
    )
    semantic_chunk_parser.add_argument(
        "query", type=str, help="Text to split into semantic (sentence-level) chunks"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk (default: 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between chunks (default: 0)",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            perform_semantic_search(args.query, args.limit)
        case "chunk":
            chunk_query(args.query, args.chunk_size, args.overlap, False)
        case "semantic_chunk":
            chunk_query(args.query, args.max_chunk_size, args.overlap, True)


if __name__ == "__main__":
    main()
