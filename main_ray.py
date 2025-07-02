import grpc
import ray
import itertools
from concurrent import futures
import os
from faster_whisper import WhisperModel

import io
import time

import proto.speech_to_text_pb2 as speech_to_text_pb2
import proto.speech_to_text_pb2_grpc as speech_to_text_pb2_grpc
from config.config import Config
from pkg.logger.logger import setup_logger
import logging
from utils.utils import record_temp

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Setup logger
setup_logger()
# Load config
Config.load()
# Init Ray
ray.init()
num_workers = 1  #estimate_ray_worker_count()
print(f"🚀  Starting {num_workers} translator workers...")
workers = [None] * num_workers
worker_cycle = None

class UnsupportedLanguageError(Exception):
    """Custom exception for unsupported language pairs."""
    pass


@ray.remote #(num_gpus=1)
class SpeechToTextWorker:
    def __init__(self):
        print(f"🚀 Loading model...")
        # self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        self.model = WhisperModel("large-v3")
        print(f"✅ Model loaded.")
    
    def transcribe_bytes(self, content: bytes,language_code:str) -> tuple[str, str]:
        """
        Transcribe speech from raw base64-encoded audio bytes.
        Returns:
            - full_text: transcription result
            - total_duration: speech duration in seconds
        """
        start_request_time = time.perf_counter()
        audio_file = io.BytesIO(content)
        segments, _info = self.model.transcribe(audio_file,language=language_code)
        full_text = "".join(segment.text for segment in segments)
        request_time = round(time.perf_counter() - start_request_time, 4)
        return full_text.strip(),f"{request_time}s"
    
    def transcribe_stream(self, file_path: str,language_code:str) -> tuple[str, str]:
        """
        Transcribe speech from raw base64-encoded audio bytes.
        Returns:
            - full_text: transcription result
            - total_duration: speech duration in seconds
        """
        start_request_time = time.perf_counter()
        segments, _info = self.model.transcribe(file_path,language=language_code,beam_size=7)
        full_text = "".join(segment.text for segment in segments)
        print(f"--->:{full_text}")
        request_time = round(time.perf_counter() - start_request_time, 4)
        return full_text.strip(),f"{request_time}s"
    
    
    
# gRPC Service definition
class SpeechToTextService(speech_to_text_pb2_grpc.SpeechToTextServiceServicer):
    def SpeechToText(self, request, context):
        global worker_cycle
        if worker_cycle is None:
            worker_cycle = itertools.cycle(workers)
        content = request.content
        language_code = request.language_code
        if language_code == "auto" or language_code == "":
            language_code = None
        try:
            worker = next(worker_cycle)
            transcript,duration = ray.get(worker.transcribe_bytes.remote(
                content,
                language_code
            ))
            return speech_to_text_pb2.SpeechToTextResponse(transcript=transcript,duration=duration)
        except Exception as e:
            logging.error(f"{e}")
            return speech_to_text_pb2.SpeechToTextResponse(error=repr(e))
    def SpeechToTextStream(self,request_iterator,context):
        global worker_cycle
        if worker_cycle is None:
            worker_cycle = itertools.cycle(workers)
        worker = next(worker_cycle)
        frames=[]
        for chunk in request_iterator:
            # chunk file 
            frames.append(chunk.content)
            print("-->")
            temp_wav = record_temp(frames)
            try:
                transcript, _ = ray.get(worker.transcribe_stream.remote(
                    temp_wav,
                    chunk.language_code,
                ))
                yield speech_to_text_pb2.SpeechToTextStreamResponse(
                    transcript=transcript or "",
                )
            except Exception as e:
                logging.error(f"{e}")
                yield speech_to_text_pb2.SpeechToTextStreamResponse(
                    transcript="",
                )
            # finally:
            #     print("final")
            #     os.remove(temp_wav)
            #     frames = []
# gRPC Server
def serve():
    global workers
    try:
        workers = [SpeechToTextWorker.remote() for _ in range(num_workers)]
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        speech_to_text_pb2_grpc.add_SpeechToTextServiceServicer_to_server(SpeechToTextService(), server)
        grpc_port = Config.get("grpc_server_port", "50052")
        server.add_insecure_port(f"[::]:{grpc_port}")
        server.start()
        print(f"✅ gRPC server started on port {grpc_port}")
        server.wait_for_termination()
    except Exception as e:
        logging.exception(f"gRPC server crashed: {e}")

if __name__ == "__main__":
    serve()
