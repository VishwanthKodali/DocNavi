# DocNavi

DocNavi is a Structured Retrieval-Augmented Generation (RAG) system designed to navigate and query complex technical documents, specifically tailored for the NASA Systems Engineering Handbook. It features a robust multi-modal ingestion pipeline and an advanced retrieval and generation query pipeline.

## Features
- **Multi-Modal Support**: Processes text, tables, and images from PDFs.
- **Hybrid Retrieval**: Combines Dense (Qdrant) and Sparse (BM25) vector retrieval with Reciprocal Rank Fusion (RRF).
- **Reranking**: Uses a cross-encoder model for high-precision result reranking.
- **Agentic Orchestration**: Uses LangGraph to orchestrate the querying process (intent routing, query expansion, reference fetching).
- **Dual UI/API**: Provides a FastAPI backend and a visually appealing NASA-themed Streamlit frontend interface.

## Quick Setup (Docker)

The easiest way to get DocNavi running is using Docker Compose, which automatically builds the FastAPI backend, the Streamlit frontend, and provisions a Qdrant vector database.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed.
- OpenAI API Key.

### Steps
1. **Configure Environment Variables**:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Build and Run the Containers**:
   ```bash
   docker-compose up --build
   ```

3. **Access the Application**:
   - **Frontend UI (Streamlit)**: `http://localhost:8501`
   - **Backend API (FastAPI)**: `http://localhost:8000`
   - **Qdrant DB**: `http://localhost:6333`

## Local Setup (Without Docker)

If you prefer to run the components locally without Docker for the application layer:

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- A running Qdrant instance.

### Steps
1. **Start Qdrant**:
   You can start just the Qdrant service using Docker:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Install Dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

3. **Set Environment Variables**:
   Update your `.env` file with necessary keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   QDRANT_URL=http://localhost:6333
   BACKEND_URL=http://localhost:8000
   ```
   *Note: Ensure `QDRANT_URL` corresponds to where Qdrant is running.*

4. **Run the Backend API**:
   ```bash
   python main.py
   ```
   *(Server starts on `localhost:8000`)*

5. **Run the Frontend UI**:
   In a separate terminal (with the virtual environment activated):
   ```bash
   streamlit run frontend.py
   ```

## Usage
1. Open the Streamlit frontend (`http://localhost:8501`).
2. Use the left sidebar to upload the NASA handbook (or another technical PDF) and click **"INGEST DOCUMENT"**.
3. Wait for the comprehensive ingestion process to finish. It will parse texts, tables, and classify/describe images.
4. Once ingested, use the main chat interface to ask technical questions. The system will retrieve relevant multi-modal context and accurately cite sources.
