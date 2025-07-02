import grpc
import proto.speech_to_text_pb2 as pb2
import proto.speech_to_text_pb2_grpc as pb2_grpc
import pyaudio
import sys
import wave

RATE = 16000
CHUNK = 1024

def has_microphone():
    p = pyaudio.PyAudio()
    print("Available audio devices:")
    has_mic = any(
        p.get_device_info_by_index(i)["maxInputChannels"] > 0
        for i in range(p.get_device_count())
    )
    p.terminate()
    return has_mic
def mic_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            print(data)
            yield pb2.SpeechToTextStreamRequest(
                content=data,
                language_code="vi"
            )
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def mic_to_wav():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    with wave.open('test_mic.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def run():
    channel = grpc.insecure_channel("localhost:50053")
    stub = pb2_grpc.SpeechToTextServiceStub(channel)
    responses = stub.SpeechToTextStream(mic_stream())
    for response in responses:
        print(f"🗣️  Text: {response.transcript} Start: {response.start_ms} End: {response.end_ms}")

if __name__ == "__main__":
    if has_microphone():
        mic_to_wav()
    else:
        print("No microphone detected. Exiting...")
        sys.exit(1)