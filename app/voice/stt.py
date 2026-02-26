from faster_whisper import WhisperModel
import os

class SpeechToText:
    def __init__(self, model_size: str = "base"):
        self.device = "cpu"
        self.compute_type = "int8" # Optimized for CPU
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        print("Whisper model loaded.")

    def transcribe(self, audio_path: str):
        if not os.path.exists(audio_path):
            print(f"STT Error: File not found at {audio_path}")
            return ""
        
        print(f"STT: Transcribing audio file at {audio_path}")
        # Added initial_prompt to help with domain-specific vocabulary
        prompt = "Data Science, Voice Agent, RAG, STT, TTS, FastAPI, Python, machine learning."
        segments, info = self.model.transcribe(audio_path, beam_size=5, initial_prompt=prompt)
        segments = list(segments) # Force evaluation
        print(f"STT: Detected language '{info.language}' with probability {info.language_probability:.1f}")
        text = " ".join([segment.text for segment in segments])
        print(f"STT Result: '{text[:50]}...'")
        return text.strip()
