from TTS.api import TTS
import os
import uuid

class TextToSpeech:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/vits"):
        self.device = "cpu"
        print(f"Loading TTS model '{model_name}' on {self.device}...")
        self.model = TTS(model_name).to(self.device)
        print("TTS model loaded.")

    def generate_audio(self, text: str, output_dir: str = "data/output_audio"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(output_dir, filename)
        
        import re
        # Clean the text: remove double parentheses or very short punctuation-only strings
        # which can crash the VITS model's sentence splitter/synthesizer
        clean_text = text.replace("()", "").strip()
        
        # If the text is empty after cleaning, return a clearer error instead of crashing
        # Use a correct alphanumeric filter (a-z, A-Z, 0-9)
        if not clean_text or len(re.sub(r'[^a-zA-Z0-9]', '', clean_text)) == 0:
            print("TTS: Skipping synthesis for empty or invalid text.")
            return None

        print(f"TTS: Starting synthesis for {len(clean_text)} characters...")
        try:
            # vits models are quite fast on CPU
            self.model.tts_to_file(text=clean_text, file_path=output_path)
            print(f"TTS: Synthesis successful. Saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"TTS Error encountered: {e}")
            import traceback
            traceback.print_exc()
            raise e
