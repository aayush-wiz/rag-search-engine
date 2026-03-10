"""Semantic search utilities using sentence-transformer embeddings and cosine similarity."""

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from cli.utils.paths import CACHE_DIR, DATA_PATH  # noqa: E402

EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_CACHE_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


class SemanticSearch:
    """Encapsulates embedding generation, storage, and cosine-similarity search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the sentence-transformer model and initialise internal state."""
        self.model = SentenceTransformer(model_name)
        self.embeddings: Optional[np.ndarray] = None
        self.documents: Optional[List[Dict[str, Any]]] = None
        self.document_map: Dict[str, Dict[str, Any]] = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        """Return the embedding vector for a single piece of text.

        Args:
            text: The string to embed. Must be non-empty.

        Returns:
            A 1-D numpy array containing the embedding.

        Raises:
            ValueError: If `text` is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Expected some text to generate the embedding.")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Encode all documents and persist the resulting embeddings to disk.

        Each document is encoded as ``"<title>: <description>"``. The embeddings
        are saved to ``EMBEDDING_CACHE_PATH`` so subsequent runs can skip
        re-encoding.

        Args:
            documents: A list of dicts, each containing at least ``id``,
                ``title``, and ``description`` keys.

        Returns:
            A 2-D numpy array of shape ``(len(documents), embedding_dim)``.
        """
        self.documents = documents
        input_texts = []
        for document in documents:
            self.document_map[document["id"]] = document
            combined_info = f"{document['title']}: {document['description']}"
            input_texts.append(combined_info)
        embeddings: np.ndarray = self.model.encode(input_texts, show_progress_bar=True)
        self.embeddings = embeddings

        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        np.save(EMBEDDING_CACHE_PATH, embeddings)

        return embeddings

    def load_or_create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Return embeddings from cache if valid, otherwise build and cache them.

        The cached file is considered valid when its length matches the number
        of documents. If the lengths differ (e.g. documents were added or
        removed), embeddings are rebuilt from scratch.

        Args:
            documents: A list of dicts, each containing at least ``id``,
                ``title``, and ``description`` keys.

        Returns:
            A 2-D numpy array of shape ``(len(documents), embedding_dim)``.
        """
        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(EMBEDDING_CACHE_PATH):
            cached_embeddings: np.ndarray = np.load(EMBEDDING_CACHE_PATH)
            if len(cached_embeddings) == len(documents):
                self.embeddings = cached_embeddings
                return cached_embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find the most semantically similar documents for a query.

        Computes cosine similarity between the query embedding and every
        document embedding, then returns the top ``limit`` results sorted by
        descending similarity score.

        Args:
            query: The search string.
            limit: Maximum number of results to return (default: 5).

        Returns:
            A list of dicts (at most ``limit`` entries), each containing:
            ``score`` (float), ``title`` (str), and ``description`` (str).

        Raises:
            ValueError: If embeddings or documents have not been loaded yet.
        """
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings or documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        results: List[Tuple[float, Dict[str, Any]]] = []

        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(doc_embedding, query_embedding)
            results.append((score, self.documents[i]))

        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[0], reverse=True)

        top_results: List[Dict[str, Any]] = []
        for score, doc in results[:limit]:
            top_results.append(
                {
                    "score": float(score),
                    "title": doc.get("title"),
                    "description": doc.get("description"),
                }
            )

        return top_results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.chunk_metadata: Optional[List[Dict[str, Any]]] = None

    def build_chunk_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        chunks_list: List[str] = []
        chunk_info: List[Dict[str, Any]] = []
        for index, document in enumerate(self.documents):
            if len(document["description"]) == 0:
                continue
            chunks = _chunk_text(document["description"], 4, 1)
            for chunk_idx, chunk in enumerate(chunks):
                chunks_list.append(chunk)
                chunk_info.append(
                    {
                        "movie_idx": index,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )
        chunk_embeddings: np.ndarray = self.model.encode(
            chunks_list, show_progress_bar=True
        )
        self.chunk_embeddings = chunk_embeddings
        self.chunk_metadata = chunk_info
        np.save(CHUNK_EMBEDDING_CACHE_PATH, chunk_embeddings)
        with open(CHUNK_METADATA_CACHE_PATH, "w") as f:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(chunks_list)},
                f,
                indent=2,
            )
        return chunk_embeddings

    def load_or_create_chunk_embeddings(
        self, documents: List[Dict[str, Any]]
    ) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(CHUNK_EMBEDDING_CACHE_PATH) and os.path.exists(
            CHUNK_METADATA_CACHE_PATH
        ):
            chunk_embeddings: np.ndarray = np.load(CHUNK_EMBEDDING_CACHE_PATH)
            with open(CHUNK_METADATA_CACHE_PATH, "r") as f:
                metadata = json.load(f)
            # Validate cache: rebuild if document count changed
            cached_doc_indices = {c["movie_idx"] for c in metadata["chunks"]}
            if len(cached_doc_indices) == len(documents):
                self.chunk_embeddings = chunk_embeddings
                self.chunk_metadata = metadata["chunks"]
                return chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        chunk_metadata = self.chunk_metadata
        if chunk_metadata is None:
            if not os.path.exists(CHUNK_METADATA_CACHE_PATH):
                raise ValueError(
                    "No chunk metadata loaded. Call `load_or_create_chunk_embeddings` first."
                )
            with open(CHUNK_METADATA_CACHE_PATH, "r") as f:
                metadata = json.load(f)
            chunk_metadata = metadata["chunks"]
            self.chunk_metadata = chunk_metadata

        chunk_embeddings = self.chunk_embeddings
        if chunk_embeddings is None:
            if not os.path.exists(CHUNK_EMBEDDING_CACHE_PATH):
                raise ValueError(
                    "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
                )
            chunk_embeddings = np.load(CHUNK_EMBEDDING_CACHE_PATH)
            self.chunk_embeddings = chunk_embeddings

        if chunk_metadata is None or chunk_embeddings is None:
            raise ValueError("Failed to load chunk embeddings or metadata.")

        if len(chunk_embeddings) != len(chunk_metadata):
            raise ValueError("Chunk embeddings and metadata are out of sync.")

        query_embedding = self.generate_embedding(query)
        chunks_score: List[Dict[str, Any]] = []
        for idx, chunk_embedding in enumerate(chunk_embeddings):
            similarity = cosine_similarity(chunk_embedding, query_embedding)
            chunk_meta = chunk_metadata[idx]
            chunks_score.append(
                {
                    "chunk_idx": chunk_meta["chunk_idx"],
                    "movie_idx": chunk_meta["movie_idx"],
                    "score": float(similarity),
                }
            )

        if self.documents is None:
            raise ValueError(
                "No documents loaded. Call `load_or_create_chunk_embeddings` first."
            )

        movie_scores: Dict[int, float] = {}
        for chunk_score in chunks_score:
            movie_idx = int(chunk_score["movie_idx"])
            score = float(chunk_score["score"])
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        ranked_movies = sorted(
            movie_scores.items(), key=lambda item: item[1], reverse=True
        )
        top_ranked_movies = ranked_movies[:limit]

        results: List[Dict[str, Any]] = []
        for movie_idx, score in top_ranked_movies:
            movie = self.documents[movie_idx]
            description = str(movie.get("description", ""))
            results.append(
                {
                    "score": score,
                    "title": movie.get("title"),
                    "description": description[:100],
                }
            )

        return results


def verify_model():
    """Print the loaded sentence-transformer model name and its max sequence length."""
    semanticsearch = SemanticSearch()
    MODEL = semanticsearch.model
    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MODEL.max_seq_length}")


def embed_text(text):
    """Generate and print the embedding for the given text.

    Displays the original text, the first 3 embedding dimensions, and the
    total number of dimensions.

    Args:
        text: The string to embed.
    """
    sementicsearch = SemanticSearch()
    embedding = sementicsearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    """Load movies from disk, build or reload their embeddings, and print a summary.

    Prints the total number of documents and the shape of the embeddings matrix.
    If the movie file is missing or malformed, an error message is printed instead.
    """
    semantic_search = SemanticSearch()
    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            movies = data.get("movies", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading movies: {e}")
        movies = []

    if not movies:
        print("No movies found to verify.")
        return

    embeddings = semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    """Generate and print the embedding vector for the given query.

    Displays the query string, the first 5 embedding dimensions, and the
    full shape of the embedding array.

    Args:
        query: The query string to embed.
    """
    semanticsearch = SemanticSearch()
    embedding = semanticsearch.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors.

    Args:
        vec1: A 1-D numpy array.
        vec2: A 1-D numpy array of the same length as ``vec1``.

    Returns:
        A float in the range [-1, 1], or 0.0 if either vector is the zero vector.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def perform_semantic_search(query: str, limit: int) -> None:
    """Load movies, run semantic search, and print the top results.

    Loads the movie dataset, builds or reloads embeddings as needed, then
    prints the top ``limit`` matches ranked by cosine similarity.

    Args:
        query: A natural-language description to search for.
        limit: Maximum number of results to display.
    """
    semantic_search = SemanticSearch()
    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            movies = data.get("movies", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading movies: {e}")
        return

    semantic_search.load_or_create_embeddings(movies)
    results = semantic_search.search(query, limit)
    for i, movie in enumerate(results, 1):
        print(f"{i}. {movie['title']} (score: {movie['score']:.4f})")
        print(f"   {movie['description']}\n")


def _chunk_text(query: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into chunks without printing. Used internally by ChunkedSemanticSearch."""
    normalized_query = query.strip()
    if not normalized_query:
        return []

    tokens = re.split(r"(?<=[.!?])\s+", normalized_query)
    if len(tokens) == 1 and not re.search(r"[.!?]\s*$", normalized_query):
        tokens = [normalized_query]

    tokens = [token.strip() for token in tokens if token.strip()]
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i : chunk_size + i]).strip()
        i += chunk_size - overlap
        if chunk:
            chunks.append(chunk)
    if len(chunks) > 1 and chunks[-1] in chunks[-2]:
        chunks.pop()
    return chunks


def chunk_query(query: str, chunk_size: int, overlap: int, semantic: bool) -> List[str]:
    """Split a query into chunks and print each one.

    In word mode (``semantic=False``) the query is split on whitespace and
    grouped into windows of ``chunk_size`` words with ``overlap`` words of
    overlap between consecutive windows.

    In semantic mode (``semantic=True``) the query is first split into
    sentences (on ``.``, ``!``, or ``?`` boundaries) and the same windowing
    logic is applied to sentences instead of words.

    Args:
        query: The text to chunk.
        chunk_size: Number of tokens (words or sentences) per chunk.
        overlap: Number of tokens shared between consecutive chunks.
        semantic: When ``True``, split on sentence boundaries; otherwise
            split on whitespace.
    """
    characters = len(query)
    chunks = _chunk_text(query, chunk_size, overlap)
    if not chunks and not query.strip():
        print(
            f"{'Semantically ' if semantic else ''}{'c' if semantic else 'C'}hunking {characters} characters"
        )
        print("1. ")
        return [""]
    print(
        f"{'Semantically ' if semantic else ''}{'c' if semantic else 'C'}hunking {characters} characters"
    )
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
    return chunks


def embed_chunk():
    chunk_semantic_search = ChunkedSemanticSearch()
    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            movies = data.get("movies", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading movies: {e}")
        movies = []

    if not movies:
        print("No movies found.")
        return
    embeddings: np.ndarray = chunk_semantic_search.load_or_create_chunk_embeddings(
        movies
    )
    print(f"Generated {len(embeddings)} chunked embeddings")


def semantic_search_chunk(query: str, limit: int):
    chunked_semantic_search = ChunkedSemanticSearch()

    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            movies = data.get("movies", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading movies: {e}")
        movies = []

    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    results = chunked_semantic_search.search_chunks(query, limit)
    for i, result in enumerate(results, 1):
        TITLE = result.get("title", "Unknown Title")
        SCORE = result.get("score", 0.0)
        DOCUMENT = result.get("description", "")
        print(f"\n{i}. {TITLE} (score: {SCORE:.4f})")
        print(f"   {DOCUMENT}...")
