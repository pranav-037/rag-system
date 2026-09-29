# RAG Pipeline Project

A complete Retrieval Augmented Generation (RAG) pipeline implementation for processing PDF documents and enabling intelligent querying capabilities.

## Overview

This project implements a comprehensive RAG pipeline that:
- Loads and processes PDF documents from a directory
- Splits documents into manageable chunks
- Generates embeddings using sentence transformers
- Stores document embeddings in a vector database (ChromaDB)
- Provides retrieval functionality for relevant document chunks
- Integrates with large language models (Groq) for generating answers
- Includes advanced features like streaming, citations, history tracking, and summarization

## Project Structure

```
.
├── data/
│   ├── pdf/                    # Source PDF documents
│   ├── text_files/             # Processed text files
│   └── vector_store/           # ChromaDB vector storage
├── notebook/
│   ├── pdf_loader.ipynb        # Main RAG pipeline implementation
│   └── document.ipynb          # Document handling examples
├── .env                        # Environment variables (API keys)
├── main.py                     # Entry point
├── pyproject.toml              # Project dependencies
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Pipeline Components

### 1. Document Ingestion
- Loads PDF files using `PyPDFLoader` and `PyMuPDFLoader`
- Supports recursive directory scanning for PDF files
- Adds metadata (source file, file type) to each document
- Handles errors gracefully during document loading

### 2. Text Processing
- Uses `RecursiveCharacterTextSplitter` for chunking documents
- Configurable chunk size (default: 1000 characters) and overlap (default: 200 characters)
- Preserves document metadata in each chunk

### 3. Embedding Generation
- Utilizes `SentenceTransformer` with the `all-MiniLM-L6-v2` model
- Generates 384-dimensional embeddings
- Includes progress tracking during embedding generation

### 4. Vector Storage
- Employs ChromaDB as the vector database
- Persistent storage for document embeddings
- Collection named `pdf_documents` with metadata description
- Efficient similarity search capabilities

### 5. Retrieval Pipeline
- `RAGRetriever` class handles query-based retrieval
- Converts queries to embeddings using the same embedding model
- Performs cosine similarity search in the vector store
- Returns ranked results with similarity scores
- Configurable top-k results and score thresholds

### 6. LLM Integration
- Integrates with Groq's `openai/gpt-oss-20b` model
- Uses temperature settings for controlled randomness
- Implements prompt engineering for context-aware responses

### 7. Advanced Features
- **Streaming responses**: Simulated streaming output for better UX
- **Citations**: Automatic citation generation with source references
- **History tracking**: Stores query history for conversation continuity
- **Summarization**: Optional answer summarization capability
- **Confidence scoring**: Provides confidence scores based on retrieval similarity

## Installation

1. Clone the repository
2. Create a virtual environment (optional but recommended)
3. Install dependencies:
   ```bash
   pip install -e .
   ```
   or
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in `.env`:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### Basic Usage (from notebooks)

Open and run the notebooks in the `notebook/` directory:
- `pdf_loader.ipynb`: Complete RAG pipeline demonstration
- `document.ipynb`: Document handling basics

### Programmatic Usage

```python
from notebook.pdf_loader import (
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    AdvancedRAGPipeline
)

# Initialize components
embedding_manager = EmbeddingManager()
vectorstore = VectorStore()
retriever = RAGRetriever(vectorstore, embedding_manager)

# For advanced features
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0.1,
    max_tokens=1024
)

advanced_rag = AdvancedRAGPipeline(retriever, llm)

# Query the system
result = advanced_rag.query(
    question="What is machine learning?",
    top_k=5,
    min_score=0.2,
    stream=True,
    summarize=True
)

print(result['answer'])
```

## Configuration

### Environment Variables
- `GROQ_API_KEY`: API key for accessing Groq's LLM services

### Vector Store Settings
- Collection name: `pdf_documents` (default)
- Persist directory: `../data/vector_store` (relative to notebook)
- Metadata description: "PDF document embeddings for RAG"

### Embedding Model
- Default: `all-MiniLM-L6-v2` (384 dimensions)
- Can be changed in `EmbeddingManager.__init__`

### Text Splitting
- Chunk size: 1000 characters (default)
- Chunk overlap: 200 characters (default)
- Separators: `["\n\n", "\n", " ", ""]`

## Features

### Core RAG Functionality
- ✅ PDF document loading and processing
- ✅ Text chunking with overlap preservation
- ✅ Embedding generation using sentence transformers
- ✅ Vector storage with ChromaDB
- ✅ Semantic search and retrieval
- ✅ LLM-powered answer generation

### Advanced Capabilities
- 🔄 Streaming response simulation
- 📖 Automatic citation generation
- 📚 Query history tracking
- 📝 Answer summarization
- 🎯 Confidence scoring for retrieved documents
- ⚙️ Configurable retrieval parameters (top-k, score thresholds)

### Error Handling
- Graceful handling of missing directories/files
- Validation of document-embedding count matching
- Exception handling for model loading and vector operations
- Informative error messages for debugging

## Dependencies

Key dependencies specified in `pyproject.toml`:
- `langchain`: Core LLM orchestration framework
- `langchain-groq`: Groq LLM integration
- `sentence-transformers`: Embedding model generation
- `chromadb`: Vector database storage
- `pypdf` / `pymupdf`: PDF processing
- `python-dotenv`: Environment variable management
- `scikit-learn`: Utility functions
- `accelerate`: Training optimizations

## Performance Notes

- Embedding dimension: 384 (all-MiniLM-L6-v2)
- Typical processing time: Seconds per document for loading, minutes for embedding generation
- Storage efficient: ChromaDB uses persistent storage with binary files
- Scalable: Can handle large document collections with proper hardware

## Future Enhancements

- [ ] Add support for additional document formats (DOCX, PPTX, etc.)
- [ ] Implement hybrid search (keyword + vector)
- [ ] Add reranking capabilities for improved relevance
- [ ] Include query expansion techniques
- [ ] Add evaluation metrics for RAG performance
- [ ] Implement caching for frequent queries
- [ ] Add REST API interface for external access
- [ ] Include metadata filtering capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Retrieval Augmented Generation (RAG) techniques
- Utilizes open-source models from Hugging Face
- Leverages LangChain framework for LLM orchestration
- Uses ChromaDB for efficient vector storage and retrieval