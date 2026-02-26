import os
import shutil
import logging
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import time
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import APIRouter, BackgroundTasks

# Setup logging to file
log_file = "app/backend.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend")

def log_debug(msg):
    print(msg)
    logger.info(msg)

from contextlib import asynccontextmanager

# Compatibility fix for coqui-tts on newer transformers/Python 3.13
import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
    transformers.pytorch_utils.isin_mps_friendly = lambda x: False

# Windows espeak-ng path fix
if os.name == 'nt':
    espeak_paths = [
        r"C:\Program Files\eSpeak NG",
        r"C:\Program Files (x86)\eSpeak NG"
    ]
    for path in espeak_paths:
        if os.path.exists(path) and path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + path
            print(f"Added {path} to PATH for espeak-ng.")

from app.rag.embeddings import EmbeddingManager
from app.rag.vector_store import VectorStore
from app.rag.generator import LLMGenerator
from app.voice.stt import SpeechToText
from app.voice.tts import TextToSpeech

# Model storage
models = {}

async def load_models_task():
    log_debug("Background: Initializing models...")
    try:
        embedding_manager = EmbeddingManager()
        
        vector_store = VectorStore(embedding_manager, logger=logger)
        vector_store.load_documents("app/data/documents")
        
        models["rag_generator"] = LLMGenerator()
        models["stt"] = SpeechToText()
        models["tts"] = TextToSpeech()
        models["vector_store"] = vector_store
        
        log_debug("Background: All models ready.")
    except Exception as e:
        log_debug(f"BACKGROUND LOAD ERROR: {e}")
        import traceback
        log_debug(traceback.format_exc())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fire off model loading in the background so health check passes immediately
    asyncio.create_task(load_models_task())
    yield
    # Cleanup if needed
    models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
api_router = APIRouter()

@api_router.middleware("http")
async def log_requests(request: Request, call_next):
    log_debug(f"Incoming: {request.method} {request.url.path}")
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    log_debug(f"Response: {response.status_code} (took {duration:.2f}s)")
    return response

@api_router.get("/health")
async def health_check():
    health = {}
    expected_models = ["rag_generator", "stt", "tts", "vector_store"]
    for name in expected_models:
        model = models.get(name)
        health[name] = "Ready" if model else "Not Loaded"
    
    overall_status = "ok" if all(models.get(name) for name in expected_models) else "degraded"
    return {"status": overall_status, "models": health}

@api_router.get("/debug")
async def debug_endpoint():
    vs = models.get("vector_store")
    return {
        "vector_store_loaded": vs is not None,
        "num_documents": len(vs.documents) if vs else 0,
        "index_ready": vs.index is not None if vs else False,
        "doc_dir_exists": os.path.exists("app/data/documents"),
        "doc_files": os.listdir("app/data/documents") if os.path.exists("app/data/documents") else []
    }

@api_router.post("/chat")
async def chat_endpoint(query: ChatQuery):
    try:
        log_debug(f"Chat request received: {query.message}")
        
        if "vector_store" not in models or not models["vector_store"]:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        if "rag_generator" not in models or not models["rag_generator"]:
            raise HTTPException(status_code=503, detail="LLM Generator not initialized")

        # 1. Search relevant documents
        context_chunks = models["vector_store"].search(query.message)
        log_debug(f"Found {len(context_chunks)} context chunks.")

        if not context_chunks:
             log_debug("No context chunks found. Proceeding with empty context.")
             # No longer returning early, so the LLM can use general knowledge.
             context_chunks = []

        # 2. Generate model response
        response_text = models["rag_generator"].generate_response(query.message, context_chunks)
        log_debug(f"Response generated: {response_text[:50]}...")
        
        return {"response": response_text}
    except Exception as e:
        log_debug(f"Error in chat: {e}")
        import traceback
        log_debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/list-documents")
async def list_documents():
    doc_dir = "app/data/documents"
    if not os.path.exists(doc_dir):
        return {"documents": []}
    files = os.listdir(doc_dir)
    return {"documents": files}

@api_router.post("/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    txn_id = str(uuid.uuid4())[:8]
    log_debug(f"[{txn_id}] UPLOAD START: {file.filename}")
    doc_dir = "app/data/documents"
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
    file_path = os.path.abspath(os.path.join(doc_dir, file.filename))
    
    try:
        content = await file.read()
        log_debug(f"[{txn_id}] Received {len(content)} bytes. Saving to {file_path}")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        log_debug(f"File saved successfully: {file_path} (Size: {len(content)} bytes)")
        
        # Verify file exists on disk
        if os.path.exists(file_path):
            log_debug(f"[{txn_id}] Verification: File exists at {file_path}")
        else:
            log_debug(f"[{txn_id}] Verification FAILED: File NOT found at {file_path} after write!")
        
        # Reload vector store to include the new PDF
        if "vector_store" in models:
            log_debug("Reloading vector store with new PDF...")
            loaded_ok = models["vector_store"].load_documents(doc_dir)
            if not loaded_ok:
                # If no text could be extracted from any documents (e.g. invalid or scanned PDF),
                # inform the user clearly instead of silently keeping an empty index.
                log_debug(f"[{txn_id}] Vector store reload found no readable text. Likely invalid or image-only PDF.")
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Uploaded PDF could not be read as text. "
                        "Please upload a proper text-based PDF (e.g., Export as PDF from Word/Docs) "
                        "instead of renaming another file to .pdf."
                    ),
                )
            log_debug("Vector store reloaded successfully.")
        else:
            log_debug("WARNING: vector_store model not found in models dict!")
        
        return {"message": f"Successfully uploaded and indexed {file.filename}"}
    except Exception as e:
        log_debug(f"CRITICAL ERROR uploading PDF: {e}")
        import traceback
        log_debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/voice-query")
async def voice_query_endpoint(file: UploadFile = File(...)):
    temp_dir = "data/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_audio_path = os.path.join(temp_dir, file.filename)
    
    try:
        log_debug(f"Voice query received: {file.filename}")
        
        # Check required models
        required = ["stt", "vector_store", "rag_generator", "tts"]
        for r in required:
            if r not in models or not models[r]:
                raise HTTPException(status_code=503, detail=f"Model '{r}' not initialized")

        # Save uploaded file
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        log_debug(f"Audio saved to {temp_audio_path}")
        
        # 1. STT: Speech to Text
        log_debug("Starting STT...")
        user_text = models["stt"].transcribe(temp_audio_path)
        if not user_text:
            log_debug("STT returned empty text.")
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        log_debug(f"Transcibed: {user_text}")
        
        # 2. RAG: Search context
        log_debug("Starting RAG search...")
        context_chunks = models["vector_store"].search(user_text)
        log_debug(f"Found {len(context_chunks)} context chunks.")
        
        # 3. LLM: Generate response text
        log_debug("Starting LLM generation...")
        response_text = models["rag_generator"].generate_response(user_text, context_chunks)
        log_debug(f"Generated text: {response_text[:50]}...")
        
        if not response_text:
            log_debug("LLM returned empty text.")
            raise HTTPException(status_code=500, detail="LLM generated an empty response.")

        # 4. TTS: Response text to Speech
        log_debug("Starting TTS...")
        output_audio_path = models["tts"].generate_audio(response_text)
        
        if not output_audio_path:
             raise HTTPException(status_code=500, detail="TTS failed to generate audio")

        log_debug(f"Audio generated at {output_audio_path}")
        
        # Return the generated WAV file
        return FileResponse(output_audio_path, media_type="audio/wav", filename="response.wav")
        
    except Exception as e:
        log_debug(f"CRITICAL ERROR in voice-query: {e}")
        import traceback
        log_debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        # Return the error message in the detail for the frontend to see
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")
    finally:
        # Simple cleanup of input audio
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# Include the API router with /api prefix
app.include_router(api_router, prefix="/api")

# Serve frontend static files
frontend_path = os.path.join(os.getcwd(), "frontend/dist")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    log_debug(f"Frontend static files mounted from {frontend_path}")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Fallback for SPA routing
    if full_path.startswith("api/"):
         raise HTTPException(status_code=404)
    
    index_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Voice Agent Backend is running."}

if __name__ == "__main__":
    import uvicorn
    # Render uses port 10000 by default
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
