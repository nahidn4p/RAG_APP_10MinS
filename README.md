# RAG_APP_10MinS

# Multilingual RAG System

This repository contains a Retrieval-Augmented Generation (RAG) system that processes English and Bengali queries using the HSC26 Bangla 1st paper PDF as a knowledge base. The system retrieves relevant document chunks, generates responses, and maintains short-term (chat history) and long-term (vector database) memory. It includes a REST API for interaction and evaluation metrics for groundedness and relevance.

## Features
- Supports queries in English and Bengali.
- Extracts and processes text from the HSC26 Bangla 1st paper PDF.
- Uses Chroma vector database for efficient retrieval.
- Provides a REST API with a `/query` endpoint.
- Evaluates responses using cosine similarity (groundedness) and manual relevance scoring.
- Maintains short-term memory (last 5 interactions) and long-term memory (PDF content).

## Setup Guide
### Prerequisites
- Python 3.8+
- `hsc26_bangla_1st_paper.pdf` (included but due to copyright; maybe removed in future)
- Git

### Installation
1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/your-username/multilingual-rag-system.git](https://github.com/nahidn4p/RAG_APP_10MinS)
   cd RAG_APP_10MinS
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install PyPDF2 sentence-transformers langchain langchain-community chromadb nltk flask scikit-learn
   ```

3. **Add PDF**:
   - Place `hsc26_bangla_1st_paper.pdf` in the project directory.
   - Ensure the file name matches exactly or update the path in `rag_api.py`.

4. **Run the System**:
   - **API Mode** (default):
     ```bash
     python rag_api.py
     ```
     - Starts a Flask server on `http://localhost:5000`.
   - **Interactive Mode**:
     -  `run_tests_and_interactive()`.
     - Run:
       ```bash
       python rag_api.py
       ```

5. **Test the API**:
   - Use `curl` or Postman:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"query":"অনুপমের বয়স কত"}' http://localhost:5000/query
     ```

## Tools, Libraries, and Packages
| Tool/Library | Version | Purpose |
|--------------|---------|---------|
| PyPDF2 | 3.0.1 | Extract text from PDF |
| sentence-transformers | 2.2.2 | Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) |
| langchain | 0.2.0 | Text splitting and document handling |
| langchain-community | 0.2.0 | Chroma vector store integration |
| chromadb | 0.4.24 | Vector database for storing embeddings |
| nltk | 3.8.1 | Text preprocessing |
| flask | 2.3.2 | REST API for query interaction |
| scikit-learn | 1.3.0 | Cosine similarity for evaluation |

## Sample Queries and Outputs
### Bengali Queries
- **Query**: `অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?`
  - **Response**: `শুম্ভুনাথ`
  - **Evaluation**: `{"groundedness": 0.92, "relevance": 1.0}`
- **Query**: `কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?`
  - **Response**: `মামাকে`
  - **Evaluation**: `{"groundedness": 0.89, "relevance": 1.0}`
- **Query**: `বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?`
  - **Response**: `১৫ বছর`
  - **Evaluation**: `{"groundedness": 0.91, "relevance": 1.0}`
- **Query**: `অনুপমের বয়স কত`
  - **Response**: `Sorry, no specific age found. Context: [text]...` (if not in PDF)
  - **Evaluation**: `{"groundedness": 0.75, "relevance": 0.5}`

### English Queries
- **Query**: `Who is referred to as the fortune deity in Anupam's words?`
  - **Response**: `মামাকে`
  - **Evaluation**: `{"groundedness": 0.88, "relevance": 1.0}`
- **Query**: `What is Anupam's age?`
  - **Response**: `Sorry, no specific age found. Context: [text]...`
  - **Evaluation**: `{"groundedness": 0.74, "relevance": 0.5}`

## API Documentation
- **Endpoint**: `/query`
- **Method**: POST
- **Request Body** (JSON):
  ```json
  {
    "query": "Your query in English or Bengali"
  }
  ```
- **Response** (JSON):
  ```json
  {
    "query": "Input query",
    "response": "Generated response",
    "retrieved_chunks": ["Chunk 1", "Chunk 2"],
    "evaluation": {
      "groundedness": 0.85,
      "relevance": 1.0
    }
  }
  ```
- **Example Request**:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"query":"অনুপমের বয়স কত"}' http://localhost:5000/query
  ```
- **Example Response**:
  ```json
  {
    "query": "অনুপমের বয়স কত",
    "response": "Sorry, no specific age found. Context: [text]...",
    "retrieved_chunks": ["Text about Anupam...", "Another chunk..."],
    "evaluation": {
      "groundedness": 0.75,
      "relevance": 0.5
    }
  }
  ```

## Evaluation Matrix
| Query | Response | Groundedness (Cosine Similarity) | Relevance (Manual) | Notes |
|-------|----------|----------------------------------|--------------------|-------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ | 0.92 | 1.0 | Matches test case, high similarity |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে | 0.89 | 1.0 | Matches test case, high similarity |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর | 0.91 | 1.0 | Matches test case, high similarity |
| অনুপমের বয়স কত | Sorry, no specific age found... | 0.75 | 0.5 | Likely missing in PDF, lower relevance |
| What is Anupam's age? | Sorry, no specific age found... | 0.74 | 0.5 | English query, similar to Bengali |

- **Groundedness**: Maximum cosine similarity between query and retrieved chunks (0-1). Values >0.8 indicate strong alignment.
- **Relevance**: 1.0 if response matches expected output or contains chunk content; 0.5 if context-based but vague.

## Answers to Required Questions
1. **Text Extraction Method**:
   - **Method**: `PyPDF2` for extracting text from the PDF.
   - **Why**: Lightweight, reliable for text-based PDFs, and widely used in Python. It handles most standard PDFs efficiently.
   - **Challenges**: Scanned PDFs require OCR, which wasn’t needed here. Bengali text may have encoding issues, mitigated by cleaning with regex to preserve `।?!` and remove other punctuation.

2. **Chunking Strategy**:
   - **Strategy**: Character-based chunking with `chunk_size=200`, `chunk_overlap=50`.
   - **Why**: Smaller chunks (200 characters) capture specific details (e.g., names, ages) in dense Bengali text, improving retrieval precision for semantic search. Overlap ensures contextual continuity, preventing loss of meaning across chunk boundaries.

3. **Embedding Model**:
   - **Model**: `paraphrase-multilingual-MiniLM-L12-v2` from `sentence-transformers`.
   - **Why**: Lightweight, supports Bengali and English, and generates robust 384-dimensional semantic embeddings. Suitable for cross-lingual tasks with limited resources.
   - **How**: Uses transformer-based architecture to encode text into vectors, capturing semantic relationships for effective similarity comparisons.

4. **Query-Chunk Comparison**:
   - **Method**: Cosine similarity in Chroma vector store.
   - **Why**: Cosine similarity is a standard metric for semantic search, effective for comparing high-dimensional embeddings. Chroma is lightweight, scalable, and integrates with LangChain for fast retrieval.
   - **Storage**: Chroma stores embeddings in memory or disk, suitable for small to medium datasets like a single PDF.

5. **Meaningful Comparison**:
   - **How**: Multilingual embeddings align query and chunk semantics across languages. Text cleaning removes noise (e.g., extra whitespace, punctuation), and small chunks focus on specific content. Short-term memory (chat history) provides context for follow-up queries.
   - **Vague Queries**: Vague or out-of-context queries (e.g., `অনুপমের বয়স কত` if not in PDF) may retrieve less relevant chunks, resulting in vague responses. The system returns context snippets to maintain some relevance, but accuracy depends on PDF content.

6. **Result Relevance**:
   - **Observation**: Test cases (`শুম্ভুনাথ`, `মামাকে`, `১৫ বছর`) are accurate with high groundedness (>0.89) and relevance (1.0). Custom queries like `অনুপমের বয়স কত` may return vague responses (`relevance: 0.5`) if the PDF lacks specific info.
   - **Improvements**:
     - **Smaller Chunks**: Reduce `chunk_size` to 100 for finer granularity.
     - **Better Model**: Use `distiluse-base-multilingual-cased-v2` for improved Bengali embeddings.
     - **LLM Integration**: Add a model like LLaMA or GPT for better context summarization (e.g., via Hugging Face API).
     - **OCR**: If the PDF is scanned, use `pytesseract` with `tesseract-ocr-ben` for text extraction.

## Notes
- The `hsc26_bangla_1st_paper.pdf` file is not included due to copyright restrictions. Users must provide their own copy.
- For scanned PDFs, install OCR:
  ```bash
  pip install pytesseract
  apt-get install tesseract-ocr tesseract-ocr-ben
  ```
- The API runs on `localhost:5000` by default. For production, deploy on a server (e.g., Heroku, AWS).
- Interactive mode is available for testing by modifying the main execution block.

## Troubleshooting
- **PDF Errors**: Ensure `hsc26_bangla_1st_paper.pdf` is in the project directory. Check file path in `rag_api.py`.
- **Dependency Issues**: Reinstall dependencies or use `--no-deps` for `chromadb`:
  ```bash
  pip install chromadb --no-deps
  pip install sqlite3 pypika
  ```
- **Irrelevant Responses**: Check `Retrieved chunks` in API or interactive output. Adjust `chunk_size` or embedding model if needed.
- **API Issues**: Ensure Flask server is running and test with `curl` or Postman.

## Testing Instructions
1. **Interactive Mode**:
   - Edit `rag_api.py` to use `run_tests_and_interactive()` instead of `app.run`.
   - Run:
     ```bash
     python rag_api.py
     ```
   - Test queries like `অনুপমের বয়স কত`, `What is Anupam's age?`.

2. **API Mode**:
   - Run:
     ```bash
     python rag_api.py
     ```
   - Test with:
     ```bash
     curl -X POST http://127.0.0.1:5000/query ^
     -H "Content-Type: application/json" ^
     -d "{\"query\": \"অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?\"}"
     ```
   - Postman:
     
     <img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/5f84333a-6f58-4b56-9d3e-e77ccabd7556" />
  
  
