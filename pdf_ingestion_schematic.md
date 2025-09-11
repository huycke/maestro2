# Architectural Schematic: PDF Ingestion Pipeline

This document provides a complete architectural schematic for the PDF ingestion pipeline in the MAESTRO application. It details the components, data flow, and technologies involved in processing a PDF file from upload to its final representation in the knowledge base.

## 1. High-Level Overview

The PDF ingestion pipeline is an asynchronous, multi-stage process designed to convert unstructured PDF documents into a structured and searchable format for the Retrieval-Augmented Generation (RAG) system. The pipeline is built on a unified database architecture, with PostgreSQL and the `pgvector` extension serving as the single source of truth for all data.

The key stages of the pipeline are:
1.  **Upload:** A PDF file is uploaded via the web interface or CLI.
2.  **Queueing:** The document is saved to disk, and a record is created in the database with a "pending" status, effectively queueing it for processing.
3.  **Processing:** A background worker picks up the queued document and initiates a multi-step processing workflow.
4.  **Parsing & Conversion:** The PDF is parsed, and its content is converted to Markdown format. This stage includes intelligent table handling.
5.  **Metadata Extraction:** Structured metadata (e.g., title, authors, abstract) is extracted from the document's content using an LLM.
6.  **Chunking:** The Markdown content is split into smaller, overlapping chunks.
7.  **Embedding:** Both dense and sparse vector embeddings are generated for each chunk using the BGE-M3 model.
8.  **Storage:** The document metadata, chunks, and embeddings are stored in the PostgreSQL database.

## 2. Architectural Diagram (Text-Based)

```
+-----------------+      +---------------------+      +-------------------------------+
|   Web UI / CLI  |----->|   FastAPI Backend   |----->|   PostgreSQL Database         |
+-----------------+      | (main.py, api/)     |      | (documents table, status: pending) |
                         +---------------------+      +-------------------------------+
                                  |                                    ^
                                  | (WebSocket Notification)           | (LISTEN/NOTIFY or Poll)
                                  v                                    |
                         +----------------------------------+          |
                         | BackgroundDocumentProcessor      |----------+
                         | (services/background_...*.py)    |
                         +----------------------------------+
                                  |
                                  v
                         +----------------------------------+
                         | DocumentProcessor                |
                         | (ai_researcher/core_rag/processor.py)|
                         +----------------------------------+
                                  |
            +---------------------+---------------------+
            | (marker library)    | (MetadataExtractor) |
            v                     v                     v
+-----------------------+  +-----------------+  +-----------------+
| PDF Parsing/Conversion|  | Metadata Extraction |  | Chunking        |
| (to Markdown)         |  | (LLM-based)     |  | (Chunker class) |
+-----------------------+  +-----------------+  +-----------------+
            |                     |                     |
            | (Markdown Content)  | (JSON Metadata)     | (Text Chunks)
            v                     v                     v
+-------------------------------------------------------------+
| TextEmbedder (BGE-M3 Model)                                 |
| (ai_researcher/core_rag/embedder.py)                        |
| - Generates dense embeddings (for semantic search)          |
| - Generates sparse embeddings (for keyword search)          |
+-------------------------------------------------------------+
                                  |
                                  v
                         +----------------------------------+
                         | PGVectorStore                    |
                         | (ai_researcher/core_rag/pgvector_store.py)|
                         +----------------------------------+
                                  |
                                  v
+----------------------------------------------------------------------+
| PostgreSQL Database (pgvector)                                       |
| - documents table: updated with metadata and status 'completed'      |
| - document_chunks table: stores chunks, dense & sparse embeddings    |
+----------------------------------------------------------------------+
```

## 3. Component Breakdown

### 3.1. FastAPI Backend

*   **Description:** The main entry point for the application. It handles API requests, user authentication, and queuing of documents for processing.
*   **Key Files:**
    *   `maestro_backend/main.py`: The main FastAPI application file. It initializes the application, includes the API routers, and starts the background processor.
    *   `maestro_backend/api/documents.py`: Defines the API endpoints for document management, including the `/api/documents/upload` endpoint that initiates the ingestion process.

### 3.2. Background Document Processor

*   **Description:** An asynchronous worker that runs in a separate thread. It listens for new documents in the queue and orchestrates the entire processing pipeline for each document.
*   **Key Files:**
    *   `maestro_backend/services/background_document_processor.py`: Contains the `BackgroundDocumentProcessor` class, which implements the main worker loop and calls the `DocumentProcessor`.

### 3.3. Document Processor

*   **Description:** The core component responsible for processing a single document. It manages the conversion, metadata extraction, chunking, and embedding stages.
*   **Key Files:**
    *   `maestro_backend/ai_researcher/core_rag/processor.py`: Contains the `DocumentProcessor` class.

### 3.4. PDF Parsing and Conversion

*   **Description:** This component is responsible for converting PDF files to Markdown. It uses the `marker` library and includes intelligent table handling.
*   **Key Libraries:**
    *   `marker-pdf-converter`: The primary library for PDF-to-Markdown conversion.
    *   `pymupdf`: Used for preliminary table detection within PDFs.
*   **Key Files:**
    *   `maestro_backend/ai_researcher/core_rag/processor.py`: The `_convert_pdf_with_table_handling` method in the `DocumentProcessor` class contains the implementation.

### 3.5. Metadata Extractor

*   **Description:** This component uses a Large Language Model (LLM) to extract structured metadata (title, authors, abstract, etc.) from the initial text of the document.
*   **Key Files:**
    *   `maestro_backend/ai_researcher/core_rag/metadata_extractor.py`: Contains the `MetadataExtractor` class.

### 3.6. Chunker

*   **Description:** This component splits the Markdown content into smaller, overlapping chunks of paragraphs.
*   **Key Files:**
    *   `maestro_backend/ai_researcher/core_rag/chunker.py`: Contains the `Chunker` class.

### 3.7. Text Embedder

*   **Description:** This component generates vector embeddings for the text chunks. It uses the `BAAI/bge-m3` model to create both dense and sparse embeddings.
*   **Key Libraries:**
    *   `FlagEmbedding`: The library used to interact with the BGE-M3 model.
*   **Key Files:**
    *   `maestro_backend/ai_researcher/core_rag/embedder.py`: Contains the `TextEmbedder` class.

### 3.8. Data Storage (PostgreSQL + pgvector)

*   **Description:** The unified data storage layer for the entire application. It uses PostgreSQL for storing structured data and the `pgvector` extension for efficient vector similarity search.
*   **Key Files:**
    *   `maestro_backend/database/models.py`: Defines the SQLAlchemy ORM models for the database tables, including `Document` and `DocumentChunk`.
    *   `maestro_backend/ai_researcher/core_rag/pgvector_store.py`: Contains the `PGVectorStore` class, which handles the storage and retrieval of vector embeddings in the `document_chunks` table.

## 4. Data Flow

1.  A user uploads a PDF file through the web UI, which sends a `POST` request to the `/api/documents/upload` endpoint in the FastAPI backend.
2.  The `upload_document` function in `api/documents.py` receives the file.
3.  The file is saved to the `/app/data/raw_files` directory.
4.  A new record is created in the `documents` table with `processing_status` set to `"pending"`.
5.  A notification is sent on the `document_queue` channel in PostgreSQL (or the background worker polls the database).
6.  The `BackgroundDocumentProcessor` receives the notification and fetches the pending document from the database.
7.  The processor updates the document's status to `"processing"`.
8.  The `DocumentProcessor` is invoked with the path to the PDF file.
9.  The `DocumentProcessor` calls the `_convert_pdf_with_table_handling` method, which uses the `marker` library to convert the PDF to Markdown text.
10. The `DocumentProcessor` uses the `MetadataExtractor` to extract structured metadata from the initial text of the document.
11. The `DocumentProcessor` uses the `Chunker` to split the Markdown text into overlapping chunks.
12. The `DocumentProcessor` uses the `TextEmbedder` to generate dense and sparse embeddings for each chunk.
13. The `DocumentProcessor` calls the `PGVectorStore` to store the chunks and their embeddings in the `document_chunks` table.
14. The `DocumentProcessor` updates the `documents` table with the extracted metadata and the final chunk count.
15. The `BackgroundDocumentProcessor` updates the document's `processing_status` to `"completed"`.

## 5. Key Technologies and Libraries

*   **Backend Framework:** FastAPI
*   **Database:** PostgreSQL
*   **Vector Search:** pgvector
*   **ORM:** SQLAlchemy
*   **PDF Parsing:** marker-pdf-converter, PyMuPDF
*   **Embedding Model:** BAAI/bge-m3 (via FlagEmbedding)
*   **Language:** Python
*   **Containerization:** Docker

## 6. Essential Components for Extraction

This section lists all the necessary components that would need to be extracted from this project to replicate the PDF ingestion pipeline.

### 6.1. Source Code Files

The following Python files contain the core logic of the ingestion pipeline and would need to be extracted and adapted:

*   **FastAPI Backend (for receiving uploads):**
    *   `maestro_backend/main.py` (for initialization logic)
    *   `maestro_backend/api/documents.py` (for the upload endpoint)
*   **Background Processing:**
    *   `maestro_backend/services/background_document_processor.py` (the main worker)
*   **Core RAG Pipeline:**
    *   `maestro_backend/ai_researcher/core_rag/processor.py` (`DocumentProcessor`)
    *   `maestro_backend/ai_researcher/core_rag/metadata_extractor.py` (`MetadataExtractor`)
    *   `maestro_backend/ai_researcher/core_rag/chunker.py` (`Chunker`)
    *   `maestro_backend/ai_researcher/core_rag/embedder.py` (`TextEmbedder`)
    *   `maestro_backend/ai_researcher/core_rag/pgvector_store.py` (`PGVectorStore`)
    *   `maestro_backend/ai_researcher/core_rag/hardware_detection.py` (for hardware acceleration)
*   **Database:**
    *   `maestro_backend/database/database.py` (for DB connection and session management)
    *   `maestro_backend/database/models.py` (for the `Document` and `DocumentChunk` models)
    *   `maestro_backend/database/crud.py` (for database operations)

### 6.2. Database Setup

*   **Database:** A PostgreSQL server is required.
*   **Extension:** The `pgvector` extension must be installed and enabled in the PostgreSQL database.
*   **Tables:** The following tables need to be created, as defined in `maestro_backend/database/models.py`:
    *   `users` (or a simplified version for authentication)
    *   `documents`
    *   `document_chunks`
*   **Initialization:** The `init_db()` function in `maestro_backend/database/database.py` can be used to create the tables and the notification trigger.

### 6.3. AI Models

*   **Embedding Model:** The `BAAI/bge-m3` model needs to be downloaded from the Hugging Face Hub. The `FlagEmbedding` library will handle this automatically.
*   **Metadata Extraction LLM:** The `MetadataExtractor` uses an LLM for its task. The specific model and API configuration need to be provided (e.g., via environment variables for OpenAI, or a local model endpoint).

### 6.4. Python Dependencies

The following key Python libraries are required. A complete list can be found in `maestro_backend/requirements.txt`.

*   `fastapi`: For the web backend.
*   `uvicorn`: For running the FastAPI server.
*   `sqlalchemy`: For database interaction.
*   `psycopg2-binary`: The PostgreSQL driver.
*   `marker-pdf-converter`: For PDF parsing.
*   `pymupdf`: For PDF table detection.
*   `FlagEmbedding`: For the BGE-M3 embedding model.
*   `torch`: The deep learning framework for the embedding model.
*   `numpy`: For numerical operations.

### 6.5. Configuration

*   **Environment Variables:** The following environment variables are essential:
    *   `DATABASE_URL`: The connection string for the PostgreSQL database.
    *   Configuration for the metadata extraction LLM (e.g., `OPENAI_API_KEY`).
*   **File System:** A directory structure for storing the raw files, converted markdown, and other artifacts needs to be created (e.g., `/app/data/raw_files`, `/app/data/processed/markdown`).
