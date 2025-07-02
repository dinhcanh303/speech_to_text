import grpc
import proto.speech_to_text_pb2 as pb2
import proto.speech_to_text_pb2_grpc as pb2_grpc

from pydub import AudioSegment
import wave

def chunk_audio(file_path, chunk_size=2):
    with wave.open(file_path, 'rb') as wf:
        framerate = wf.getframerate()
        # frames_per_chunk = framerate * chunk_size
        while True:
            # data = wf.readframes(chunk_size)
            data = wf.readframes(framerate)
            if not data:
                break
            yield data
def chunk_audio_pydub(file_path, chunk_size=2):
    audio = AudioSegment.from_file(file_path,format="mp3")
    frame_rate = audio.frame_rate
    chunk_length_ms = chunk_size * 1000
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        yield chunk.raw_data
def run():
    channel = grpc.insecure_channel("localhost:50053")
    stub = pb2_grpc.SpeechToTextServiceStub(channel)
    def request_generator():
        for chunk in chunk_audio_pydub("warm_up_file.wav"):
            yield pb2.SpeechToTextStreamRequest(
                content=chunk,
                language_code="auto"
            )
    # Call RPC Stream
    responses = stub.SpeechToTextStream(request_generator())
    for response in responses:
        print(f"🗣️  Text: {response.transcript} Start: {response.start_ms} End: {response.end_ms}")

if __name__ == "__main__":
    run()
