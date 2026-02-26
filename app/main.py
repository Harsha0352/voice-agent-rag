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
from fastapi import APIRouter

# Setup logging to stdout for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend")

def log_debug(msg):
    logger.info(msg)

from contextlib import asynccontextmanager

# Compatibility fix for coqui-tts on newer transformers/Python 3.13
try:
    import transformers.pytorch_utils
    if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
        transformers.pytorch_utils.isin_mps_friendly = lambda x: False
except ImportError:
    pass

# Defer heavy imports to background loading
EmbeddingManager = None
VectorStore = None
LLMGenerator = None
SpeechToText = None
TextToSpeech = None

# Model storage
models = {}

async def load_models_task():
    global EmbeddingManager, VectorStore, LLMGenerator, SpeechToText, TextToSpeech
    log_debug("Background: Importing heavy models...")
    try:
        from app.rag.embeddings import EmbeddingManager as EM
        from app.rag.vector_store import VectorStore as VS
        from app.rag.generator import LLMGenerator as LG
        from app.voice.stt import SpeechToText as STT
        from app.voice.tts import TextToSpeech as TTS
        
        EmbeddingManager, VectorStore, LLMGenerator, SpeechToText, TextToSpeech = EM, VS, LG, STT, TTS
        
        log_debug("Background: Initializing model instances...")
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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    log_debug(f"Incoming: {request.method} {request.url.path}")
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    log_debug(f"Response: {response.status_code} (took {duration:.2f}s)")
    return response

# Create API router
api_router = APIRouter()

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

class ChatQuery(BaseModel):
    message: str

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
        
        # Reload vector store to include the new PDF
        if "vector_store" in models:
            log_debug("Reloading vector store with new PDF...")
            loaded_ok = models["vector_store"].load_documents(doc_dir)
            if not loaded_ok:
                log_debug(f"[{txn_id}] Vector store reload found no readable text.")
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded PDF could not be read as text.",
                )
            log_debug("Vector store reloaded successfully.")
        
        return {"message": f"Successfully uploaded and indexed {file.filename}"}
    except Exception as e:
        log_debug(f"CRITICAL ERROR uploading PDF: {e}")
        if isinstance(e, HTTPException):
            raise e
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
        
        # 1. STT
        user_text = models["stt"].transcribe(temp_audio_path)
        if not user_text:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # 2. RAG
        context_chunks = models["vector_store"].search(user_text)
        
        # 3. LLM
        response_text = models["rag_generator"].generate_response(user_text, context_chunks)
        
        # 4. TTS
        output_audio_path = models["tts"].generate_audio(response_text)
        
        if not output_audio_path:
             raise HTTPException(status_code=500, detail="TTS failed to generate audio")

        return FileResponse(output_audio_path, media_type="audio/wav", filename="response.wav")
        
    except Exception as e:
        log_debug(f"CRITICAL ERROR in voice-query: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# Include API router
app.include_router(api_router, prefix="/api")

# Serve frontend
frontend_path = os.path.join(os.getcwd(), "frontend/dist")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    
    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        index_file = os.path.join(frontend_path, "index.html")
        return FileResponse(index_file)
else:
    @app.get("/")
    async def root():
        return {"message": "Voice Agent Backend is running. Frontend missing."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
