import grpc
import ray
import itertools
from concurrent import futures
import os
import whisper
from typing import Any, List, Optional, Tuple

from proto import speech_to_text_pb2 , speech_to_text_pb2_grpc
from config.config import Config
from pkg.logger.logger import setup_logger
import sys

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Setup logger
logger = setup_logger(__name__)
# Load config
Config.load()
# Init Ray
ray.init()
num_workers = 3  #estimate_ray_worker_count()
print(f"🚀  Starting {num_workers} translator workers...")
workers = [None] * num_workers
worker_cycle = None

class UnsupportedLanguageError(Exception):
    """Custom exception for unsupported language pairs."""
    pass

@ray.remote(num_gpus=1/3)
class WhisperWorker:
    def __init__(self, args: Any):
        print(f"🚀 Loading model...")
        self.args = args
        self.asr, self.online = whisper.asr_factory(args)
        self.min_chunk = args.min_chunk_size
        self.is_first = True
        self.buffer: List[Any] = []
        self.SAMPLING_RATE = 16000
        #warmup file
        msg = "Whisper is not warmed up. The first chunk processing may take longer."
        if args.warmup_file:
            if os.path.isfile(args.warmup_file):
                a = whisper.load_audio_chunk(args.warmup_file,0,1)
                _ = self.asr.transcribe(a)
                logger.info("Whisper is warmed up.")
            else:
                logger.critical("The warm up file is not available. "+msg)
                sys.exit(1)
        else:
            logger.warning(msg)

    def process_init(self) -> None:
        self.online.init()

    def clear(self):
        self.buffer = []

    def receive_audio_chunk(self,chunk_bytes: bytes) -> bool:
        import soundfile, io, librosa, numpy as np

        sf = soundfile.SoundFile(io.BytesIO(chunk_bytes),
                                 channels=1,
                                 endian="LITTLE",
                                 samplerate=self.SAMPLING_RATE,
                                 subtype="PCM_16",
                                 format="RAW")
        audio, _ = librosa.load(sf, sr=self.SAMPLING_RATE, dtype=np.float32)
        self.buffer.append(audio)
        total_samples = sum(len(x) for x in self.buffer)
        return total_samples >= self.min_chunk * self.SAMPLING_RATE

    def process_chunk(self) -> Tuple[int, int, str]:
        import numpy as np
        audio = np.concatenate(self.buffer)
        self.buffer.clear()
        self.online.insert_audio_chunk(audio)
        return self.online.process_iter()
    
    def transcribe_bytes(self, content: bytes,language:str) -> tuple[str, str]:
        """
        Transcribe speech from raw base64-encoded audio bytes.
        Returns:
            - full_text: transcription result
            - total_duration: speech duration in seconds
        """
        return self.asr.transcribe_bytes(content,language)
    
# gRPC Service definition
class WhisperService(speech_to_text_pb2_grpc.SpeechToTextServiceServicer):
    def __init__(self,args: Any,workers: List[Any]):
        self.args = args
        self.workers = workers
        self.worker_cycle = itertools.cycle(workers)

    def SpeechToText(self, request, context):
        content = request.content
        language_code = request.language_code
        if language_code == "auto" or language_code == "":
            language_code = None
        try:
            worker = next(self.worker_cycle)
            transcript,duration = ray.get(worker.transcribe_bytes.remote(
                content,
                language_code
            ))
            return speech_to_text_pb2.SpeechToTextResponse(transcript=transcript,duration=duration)
        except Exception as e:
            logger.error(f"{e}")
            context.abort(grpc.StatusCode.INTERNAL, repr(e)) 
    def SpeechToTextStream(self,request_iterator,context):
        try:
            worker = next(self.worker_cycle)
            for chunk in request_iterator:
                enough = worker.receive_audio_chunk.remote(chunk.content)
                if enough:
                    beg, end, text = ray.get(worker.process_chunk.remote())
                    yield speech_to_text_pb2.SpeechToTextStreamResponse(
                        start_ms=f"{beg}",
                        end_ms=f"{end}",
                        transcript=text or "",
                    )
            # Flush any remaining audio
            _ = worker.receive_audio_chunk.remote(b"")  # Optionally flush
            beg, end, text = ray.get(worker.process_chunk.remote())
            if text:
                yield speech_to_text_pb2.SpeechToTextStreamResponse(
                    start_ms=f"{beg}",
                    end_ms=f"{end}",
                    transcript=text,
                )
            worker.clear.remote()

        except Exception as e:
            logger.error(f"{e}")
            context.abort(grpc.StatusCode.INTERNAL, repr(e)) 
# gRPC Server
def serve(args: Any) -> None:
    global workers
    try:
        workers = [WhisperWorker.remote(args) for _ in range(num_workers)]
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        speech_to_text_pb2_grpc.add_SpeechToTextServiceServicer_to_server(WhisperService(args, workers), server)
        grpc_port = Config.get("grpc_server_port", "50053")
        server.add_insecure_port(f"[::]:{grpc_port}")
        server.start()
        print(f"✅ gRPC server started on port {grpc_port}")
        server.wait_for_termination()
    except Exception as e:
        logger.exception(f"gRPC server crashed: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-file",
                        type=str,
                        default="warm_up_file.wav", 
                        dest="warmup_file", 
                        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")
    whisper.add_shared_args(parser)
    args = parser.parse_args()
    serve(args)
