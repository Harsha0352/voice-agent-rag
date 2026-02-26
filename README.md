# Voice Agent with RAG (Open-Source)

This is a fully open-source Voice Agent with Retrieval-Augmented Generation (RAG) built using Python and FastAPI. It is designed to run on a CPU-only environment.

## Features
- **Speech-to-Text (STT)**: Powered by `faster-whisper`.
- **RAG**: Combines `sentence-transformers` for embeddings and `FAISS` for vector storage.
- **LLM**: Uses `microsoft/phi-2` for response generation (CPU-focused).
- **Text-to-Speech (TTS)**: Powered by `Coqui TTS` (VITS models).

## Project Structure
```text
app/
├── main.py            # FastAPI entry point & model orchestration
├── rag/
│   ├── embeddings.py  # Embedding generation logic
│   ├── vector_store.py # FAISS index management
│   └── generator.py   # LLM generation logic
├── voice/
│   ├── stt.py         # Speech-to-Text conversion
│   └── tts.py         # Text-to-Speech conversion
└── data/
    └── documents/     # Place your .txt knowledge base here
```

## Setup Instructions

### 1. Prerequisites
- Python 3.10 or 3.11 recommended.
- FFmpeg installed on your system (required for `faster-whisper`).

### 2. Environment Setup
```powershell
# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Application
```powershell
uvicorn app.main:app --reload
```
The first run will download several GBs of models from Hugging Face. Ensure you have a stable internet connection.

## API Documentation

### POST `/chat`
Text-only RAG inquiry.
- **Body**: `{"message": "How does this project work?"}`
- **Response**: `{"response": "..."}`

### POST `/voice-query`
Full voice-to-voice pipeline.
- **Form-Data**: `file` (audio file like .wav or .mp3)
- **Response**: Returns a `.wav` file with the generated response.

## Example Request (curl)

**Chat Endpoint:**
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What models does this use?"}'
```

**Voice Query Endpoint:**
```bash
curl -X POST "http://localhost:8000/voice-query" \
     -F "file=@path/to/your/audio.wav" \
     --output response.wav
```

## How Models are Loaded
Models are loaded once during application startup using the FastAPI `lifespan` event. This ensures that expensive resources (LLM, Whisper, TTS) are kept in memory and reused across requests, preventing crashes and slow response times.
