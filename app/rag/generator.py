import logging
import requests
import os
from huggingface_hub import InferenceClient

logger = logging.getLogger("backend")

class LLMGenerator:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", token: str = None):
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            logger.error("LLM: HF_TOKEN environment variable not found!")
        else:
            logger.info(f"LLM: HF_TOKEN detected (starts with {self.token[:5]}...)")
        
        self.model_name = model_name
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        logger.info(f"LLM: Initialized for model '{model_name}' using endpoint '{self.api_url}'.")

    def generate_response(self, query: str, context_chunks: list[str]):
        if not self.token:
            raise RuntimeError("Hugging Face Token (HF_TOKEN) is missing. Please set it in your .env file.")

        context = "\n".join(context_chunks)
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Use the provided Context to answer the question if possible. If the context is empty or doesn't contain the answer, you may use your general knowledge, but clearly state that the answer is based on general knowledge and not the uploaded document. Be brief and direct."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ],
            "max_tokens": 512,
            "temperature": 0.1
        }
        
        print(f"Generating response via HF Router API ({self.model_name})...")
        try:
            res = requests.post(self.api_url, headers=headers, json=payload, timeout=20)
            
            if res.status_code != 200:
                # Handle specific errors
                error_data = res.json() if res.headers.get("content-type") == "application/json" else {"error": res.text}
                raise RuntimeError(f"HF API returned {res.status_code}: {error_data.get('error', res.text)}")
            
            result = res.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            
            raise RuntimeError(f"Unexpected API response format: {result}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"LLM Connection failed: {error_msg}")
            
            # Very basic fallback for non-chat models if needed, but router v1 is standard now
            raise RuntimeError(f"LLM Connection failed: {error_msg}")
