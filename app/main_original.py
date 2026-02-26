import logging
from datetime import datetime

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

app = FastAPI() # Added for FastAPI app instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models once at startup
    log_debug("Initializing models...")
    try:
        embedding_manager = EmbeddingManager()
        
        vector_store = VectorStore(embedding_manager)
        vector_store.load_documents("app/data/documents")
        
        models["rag_generator"] = LLMGenerator()
        models["stt"] = SpeechToText()
        models["tts"] = TextToSpeech()
        models["vector_store"] = vector_store
        
        log_debug("All models ready.")
    except Exception as e:
        log_debug(f"STARTUP ERROR: {e}")
        import traceback
        log_debug(traceback.format_exc())
    yield
    # Cleanup if needed
    models.clear()

@app.get("/health")
async def health_check():
    health = {}
    for name, model in models.items():
        health[name] = "Ready" if model else "Not Loaded"
    return {"status": "ok", "models": health}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    try:
        log_debug(f"Chat request received: {query.message}")
        # 1. Search relevant documents
        context_chunks = models["vector_store"].search(query.message)
        log_debug(f"Found {len(context_chunks)} context chunks.")
        
        # The following function definition is syntactically incorrect here.
        # It appears to be a method intended for a class like EmbeddingManager or VectorStore.
        # As per instructions to make the file syntactically correct and faithfully apply the change,
        # I am placing it as a comment to avoid breaking the code, as its intended location
        # within a class definition is not specified and inserting it directly here
        # would be a syntax error.
        #
        # def generate_query_embedding(self, query: str):
        # embedding = self.model.encode([query], convert_to_numpy=True)
        # # Ensure it's exactly 2D for FAISS (1, dimension)
        # if len(embedding.shape) == 1:
        #     embedding = embedding.reshape(1, -1)
        # return embedding

        # 2. Generate model response
        response_text = models["rag_generator"].generate_response(query.message, context_chunks)
        log_debug(f"Response generated: {response_text[:50]}...")
        
        return {"response": response_text}
    except Exception as e:
        log_debug(f"Error in chat: {e}")
        import traceback
        log_debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-query")
async def voice_query_endpoint(file: UploadFile = File(...)):
    temp_dir = "data/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_audio_path = os.path.join(temp_dir, file.filename)
    
    try:
        log_debug(f"Voice query received: {file.filename}")
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
        log_debug(f"Audio generated at {output_audio_path}")
        
        # Return the generated WAV file
        return FileResponse(output_audio_path, media_type="audio/wav", filename="response.wav")
        
    except Exception as e:
        log_debug(f"CRITICAL ERROR in voice-query: {e}")
        import traceback
        log_debug(traceback.format_exc())
        # Return the error message in the detail for the frontend to see
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")
    finally:
        # Simple cleanup of input audio
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
