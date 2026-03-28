# Project details: DocNavi

DocNavi is an advanced Structured Retrieval-Augmented Generation (RAG) system built to parse, structure, and accurately answer questions over complex technical documents. It has been primarily designed to ingest and query the NASA Systems Engineering Handbook (SP-2016-6105 Rev2).

## Architecture Overview

The project is split into two primary components: the **Ingestion Pipeline** and the **Query Pipeline**, both orchestrated via a FastAPI backend, and interacted with through a Streamlit frontend UI.

### 1. Ingestion Pipeline
The ingestion pipeline processes PDFs synchronously and asynchronously to ensure efficient and non-blocking multi-modal parsing.

- **PDF Parsing**: Synchronously extracts blocks of information (text, images, and tables) from PDF pages.
- **Content Classification**: Categorizes blocks and applies hierarchical chunking to maintain section boundaries and document context.
- **Table Extraction & VLM Descriptions**: Runs concurrently. Extracts structured representations of tables while sending extracted images to OpenAI's GPT-4o vision model for detailed text descriptions.
- **Acronym Resolution**: Automatically resolves acronyms found in the text to assist in better query expansions later.
- **Multi-Collection Embedding**: Embeds and upserts data into **three distinct Qdrant collections**:
  - `text_chunks`: Standard paragraph text chunks.
  - `image_store`: VLM-generated descriptions mapped to their source images.
  - `table_store`: Serialized table data.
- **Sparse Indexing**: Builds a BM25 index over the text chunks for precise keyword retrieval as a complement to vector similarity search.

### 2. Query Pipeline (LangGraph Orchestration)
The query execution is orchestrated dynamically using a LangGraph `StateGraph`, enabling sophisticated, step-by-step reasoning and retrieval:

1. **Intent Router**: Uses a lightweight LLM call to determine if the query requires handbook retrieval or if it's a general/off-topic question that can be answered directly.
2. **Query Expansion**: Resolves acronyms in the user query to align with document terminology.
3. **Parallel Retrieval**: Executes a dense embedding search (Qdrant) and a sparse keyword search (BM25) in parallel.
4. **RRF Fusion**: Combines dense and sparse results using Reciprocal Rank Fusion to balance semantic meaning and exact keyword matches.
5. **Reranker**: Employs a local cross-encoder model (`ms-marco-MiniLM-L-6-v2`) to accurately rerank the top fused candidates against the initial query.
6. **Reference Detector & Augmentation**: Analyzes the top-ranked chunks. If a chunk mentions an image, table, or cross-referenced section, those artifacts are fetched from the respective Qdrant collections and injected into the context.
7. **Context Assembler**: Composes a heavily grounded context string with proper source metadata (Section IDs, page numbers, extracted diagrams, and tables).
8. **Generator**: Uses GPT-4o with strict system prompts to answer the query *only* based on the retrieved context, forcing accurate source citations.
9. **Citation Validator**: A secondary check ensuring that any sections cited in the generated answer were indeed provided in the context, preventing hallucinations.

## Technology Stack

- **Application Framework**: FastAPI (Backend API), Streamlit (Frontend interface).
- **Core Orchestration**: LangGraph, for stateful query reasoning.
- **LLM Capabilities**:
  - Generation & Intent Routing: `gpt-4o`
  - Vision (Images): `gpt-4o`
  - Embeddings: `text-embedding-3-small` (1536 dims)
- **Vector Database**: Qdrant (for handling multiple collections).
- **Keyword Search**: `rank-bm25` (Sparse retrieval).
- **Reranker**: `sentence-transformers` (Cross-Encoder).
- **Package Management**: `uv`.
- **Infrastructure**: Docker & Docker Compose for rapid spin-up.

## Application Structure
- `/src/app/API.py`: Main FastAPI application, lifespan handling, and routers inclusion.
- `/src/app/components/`: Contains the core logic for the ingestion steps (parsers, descriptors, embedders) and the query steps (LangGraph nodes).
- `/src/app/database/`: Qdrant client connection and initialization scripts.
- `/src/app/routers/`: FastAPI route definitions (`/ingest`, `/query`, etc.).
- `frontend.py`: The cohesive NASA-styled dark-mode Streamlit UI.
- `docker-compose.yml` & `Dockerfile`: Multi-service application hosting definitions.
- `config.yml`: Global hyperparameters, models, and directory definitions.
