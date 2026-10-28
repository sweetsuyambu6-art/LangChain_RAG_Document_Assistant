# LangChain RAG Document Assistant

A portfolio project that uses **LangChain** to build a Retrieval-Augmented Generation (RAG) question-answering app over user-uploaded documents.

## Tech stack
- **LangChain**: document processing workflow, prompt template, and runnable chain
- **Streamlit**: user interface
- **Chroma**: vector database and semantic retrieval
- **Hugging Face Sentence Transformers**: text embeddings
- **Ollama**: local chat model
- **PyPDF / Docx2txt**: document loading

## RAG flow
1. Load PDF, DOCX, or TXT documents with LangChain document loaders.
2. Split the text into overlapping chunks with `RecursiveCharacterTextSplitter`.
3. Embed the chunks with `HuggingFaceEmbeddings`.
4. Store vectors in Chroma.
5. Retrieve the top-k chunks relevant to the user's question.
6. Pass the question and retrieved context to `ChatOllama` through a `ChatPromptTemplate` chain.
7. Show the answer and retrieved source chunks for manual verification.

## Run locally
Requirements: Python 3.10+ and Ollama installed.

```bash
python -m venv .venv
```

Activate the environment:
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages:

```bash
pip install -r requirements.txt
```

Download the chat model and ensure Ollama is running:

```bash
ollama pull llama3.2
```

Copy `.env.example` to `.env` if you want to configure `OLLAMA_MODEL`. Then launch:

```bash
streamlit run app.py
```

Upload documents, click **Index documents**, and ask questions. The first run downloads the embedding model and may take a while.

## Evaluation
Create questions with known answers and source passages. Check retrieval relevance, answer faithfulness, answer completeness, ability to say “not found,” and response latency. Record measured results before reporting metrics.

## Use after you have run and tested it
**LangChain RAG Document Assistant | Python, LangChain, Chroma, Hugging Face, Ollama, Streamlit**
- Built a document question-answering app using LangChain to retrieve relevant passages from PDF, DOCX, and TXT files.
- Implemented text splitting, embedding generation, and Chroma vector retrieval for semantic search.
- Connected retrieved context to an Ollama chat model using a LangChain prompt and runnable chain.
- Added source-passage inspection to help users verify answers against document evidence.

## Limitations
- Scanned PDFs need OCR, not included here.
- Vector index exists for the current Streamlit session; rebuild after restarting.
- This is a starter project; add authentication, upload size limits, logging, and privacy controls before public deployment.
- Do not claim accuracy or latency metrics until measured.
