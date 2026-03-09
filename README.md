# RAG Search Engine

A hybrid search engine for movie data that combines traditional keyword-based search (BM25, TF-IDF) with modern semantic search using sentence transformers.

## Features

- **Keyword Search**: Traditional information retrieval using BM25 and TF-IDF algorithms
- **Semantic Search**: Vector-based search using sentence embeddings and cosine similarity
- **Inverted Index**: Efficient document indexing for fast keyword lookups
- **Caching**: Persistent storage of indices and embeddings for quick reloads

## Project Structure

```
rag-search-engine/
├── cli/
│   ├── keyword_search_cli.py    # CLI for keyword/BM25 search
│   ├── semantic_search_cli.py   # CLI for semantic search
│   └── lib/
│       └── semantic_search.py   # Semantic search implementation
├── lib/
│   └── keyword_search.py        # Keyword search & indexing logic
├── utils/
│   └── search_utils.py          # Shared utilities (tokenization, stemming)
├── data/
│   ├── movies.json              # Movie dataset
│   └── stopwords.txt            # Stop words for text processing
└── cache/                       # Cached indices and embeddings
```

## Installation

Requires Python 3.12+

```bash
# Install dependencies
pip install -e .
```

Or using uv:

```bash
uv sync
```

## Usage

### Keyword Search CLI

First, build the index:

```bash
python cli/keyword_search_cli.py build
```

Available commands:

| Command | Description | Example |
|---------|-------------|---------|
| `search` | Basic keyword search | `python cli/keyword_search_cli.py search "star wars"` |
| `tf` | Get term frequency | `python cli/keyword_search_cli.py tf 1 "jedi"` |
| `idf` | Get IDF score | `python cli/keyword_search_cli.py idf "jedi"` |
| `tfidf` | Get TF-IDF score | `python cli/keyword_search_cli.py tfidf 1 "jedi"` |
| `bm25idf` | Get BM25 IDF | `python cli/keyword_search_cli.py bm25idf "jedi"` |
| `bm25tf` | Get BM25 TF | `python cli/keyword_search_cli.py bm25tf 1 "jedi"` |
| `bm25search` | Full BM25 search | `python cli/keyword_search_cli.py bm25search "space adventure" --limit 5` |

### Semantic Search CLI

Available commands:

| Command | Description | Example |
|---------|-------------|---------|
| `verify` | Verify model | `python cli/semantic_search_cli.py verify` |
| `embed_text` | Generate embedding | `python cli/semantic_search_cli.py embed_text "query text"` |
| `verify_embeddings` | Check embeddings | `python cli/semantic_search_cli.py verify_embeddings` |
| `embedquery` | Vector from query | `python cli/semantic_search_cli.py embedquery "query"` |
| `search` | Semantic search | `python cli/semantic_search_cli.py search "space adventure" --limit 5` |

## Algorithms

### BM25

The BM25 ranking function uses:
- **BM25 IDF**: `log((N - DF + 0.5) / (DF + 0.5) + 1)`
- **BM25 TF**: `(tf * (k1 + 1)) / (tf + k1 * ((1 - b) + b * (doc_len / avg_dl)))`

Default parameters: `k1=1.5`, `b=0.75`

### TF-IDF

- **TF**: Raw term frequency in a document
- **IDF**: `log((total_docs + 1) / (matching_docs + 1))`

### Semantic Search

Uses `sentence-transformers/all-MiniLM-L6-v2` model to generate 384-dimensional embeddings and cosine similarity for ranking.

## Data Format

The movie dataset (`data/movies.json`) should follow this structure:

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Movie Title",
      "description": "Movie description..."
    }
  ]
}
```

## Dependencies

- `nltk` - Natural language processing (stemming)
- `numpy` - Numerical operations
- `sentence-transformers` - Text embeddings

## License

MIT
