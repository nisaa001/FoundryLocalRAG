Offline RAG Assistant with Microsoft Foundry Local

A local Retrieval-Augmented Generation (RAG) project developed with Microsoft Foundry Local, Python, and Streamlit.

The application is designed to read PDF documents, split their content into chunks, generate vector embeddings, retrieve the most relevant sections, and produce answers using a locally running language model.

Features

Fully local model inference

PDF text extraction

Document chunking

Semantic search

Vector embedding generation

Retrieval-Augmented Generation workflow

Streamlit-based user interface

Local testing scripts for embedding and chat models

Technologies

Python

Streamlit

Microsoft Foundry Local

Phi-4 Mini

Qwen3 Embedding 0.6B

OpenAI-compatible SDK

PDF processing libraries

How It Works

A PDF document is loaded from the data folder.

The document text is extracted and divided into smaller chunks.

Qwen3 Embedding generates a vector representation for each chunk.

The user's question is converted into an embedding.

The most relevant document chunks are retrieved through semantic similarity.

Phi-4 Mini uses the retrieved context to generate an answer.

The result is displayed through the Streamlit interface.

Project Structure

FoundryLocalRAG/
├── data/
│   ├── bilgi.txt
│   └── foundry_local_plan.pdf
├── app.py
├── embedding_test.py
├── pdf_reader.py
├── rag_app.py
├── rag_app_working.py
├── streamlit_app.py
├── vector_search.py
├── test_chat_dual.py
├── test_chat_long.py
├── test_chat_many.py
├── test_streamlit_dual.py
├── test_streamlit_minimal.py
└── testchat.py

Requirements

Before running the project, install:

Python 3.10 or later

Microsoft Foundry Local

Phi-4 Mini model

Qwen3 Embedding 0.6B model

Create and activate a virtual environment:

Windows PowerShell

python -m venv .venv
.venv\Scripts\Activate.ps1

Install the required Python packages:

pip install -r requirements.txt

Note: A requirements.txt file should be included in the repository before sharing the project for reproducible installation.

Running the Application

Start Microsoft Foundry Local and make sure the required models are available.

Then run:

streamlit run streamlit_app.py

Open the local address displayed in the terminal, usually:

http://localhost:8501

Model Tests

To verify that the local embedding and chat models are working, run the available test files individually, for example:

python embedding_test.py
python test_chat_dual.py
streamlit run test_streamlit_dual.py

Current Status

The local embedding model and Phi-4 Mini chat model have been tested successfully.

The full Streamlit RAG workflow is still under development. In some RAG runs, the local chat-completion request may return an Operation was cancelled error. The repository includes multiple test files used to isolate and investigate this model-lifecycle issue.

Planned Improvements

Resolve the remaining RAG chat-completion cancellation issue

Add support for uploading PDFs through the interface

Add multi-document retrieval

Add chat history

Improve document chunking

Add source references to generated answers

Organize test files under a dedicated tests folder

Add automated setup instructions

Author

Nisanur GündüzSoftware Engineering Student

Repository

GitHub: https://github.com/nisaa001/FoundryLocalRAG
