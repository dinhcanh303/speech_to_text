import os
import ray
from ray import serve
from fastapi import FastAPI, UploadFile, Form, HTTPException
import argparse
import whisper
import time
from typing import Optional
from pydub import AudioSegment
from io import BytesIO
from config.config import Config
from pkg.logger.logger import setup_logger
# ----------------------------------------
rest_port = Config.get("rest_server_port", "50053")
# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--warmup-file", type=str,
                    default="warm_up_file.wav", 
                    dest="warmup_file")
whisper.add_shared_args(parser)
args = parser.parse_args()
logger = setup_logger(__name__)
# ----------------------------------------
# Init Ray Serve
ray.init()
serve.start(http_options={"host": "0.0.0.0","port": rest_port})
app = FastAPI()


# ----------------------------------------
class WhisperWorker:
    def __init__(self, args):
        self.asr, _ = whisper.asr_factory(args)
        if args.warmup_file and os.path.exists(args.warmup_file):
            print("🔥 Warming up...")
            audio = whisper.load_audio_chunk(args.warmup_file, 0, 1)
            self.asr.transcribe(audio)
        print("✅ Whisper ready")

    def transcribe_bytes(self, content: bytes, language: str):
        return self.asr.transcribe_bytes(content, language)
# ----------------------------------------
@serve.deployment( num_replicas=3,ray_actor_options={"num_gpus": 1/3})
@serve.ingress(app)
class WhisperDeployment:
    def __init__(self):
        self.worker = WhisperWorker(args)
        print("✅ WhisperDeployment ready")

    @app.post("/speech-to-text")
    async def transcribe(self, file: UploadFile, language: Optional[str] = Form(None)):
        if not language or language.lower() == "auto":
            language = None
        # check file extension
        allowed_exts = [".mp3", ".m4a"]
        filename = file.filename.lower()
        if not any(filename.endswith(ext) for ext in allowed_exts):
            raise HTTPException(status_code=400, detail="Only .mp3 or .m4a files are allowed.")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file or upload failed.")
        # Using pydub calculation duration
        try:
            audio = AudioSegment.from_file(BytesIO(content))
            duration_sec = audio.duration_seconds
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")
        # Check duration
        if duration_sec > 60:
            raise HTTPException(status_code=400, detail="Audio must be shorter than 1 minute.")
        
        transcript, duration = self.worker.transcribe_bytes(content, language)
        return {"transcript": transcript, "duration": duration}

# ----------------------------------------
serve.run(WhisperDeployment.bind(),route_prefix="/v1")
print(f"✅ Ray Serve FastAPI running at http://0.0.0.0:{rest_port}/v1/speech-to-text")
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("⏹️ Stopped by user.")

