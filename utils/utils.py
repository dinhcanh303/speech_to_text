import torch
from common.constant import model_memory_usage
import math
import tempfile
import wave
# get gpu memory
def get_gpu_memory():
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device_id).total_memory // (1024 ** 2)  # MB
        return total_memory
    return 0

def estimate_ray_worker_count(per_worker_vram=model_memory_usage):  # MB
    total_gpu_memory = get_gpu_memory()
    if total_gpu_memory == 0:
        print("❌ No GPU detected. Using CPU only.")
        return 1
    usable_memory = int(total_gpu_memory * 0.9)  # 10% buffer
    worker_count = max(1, usable_memory // per_worker_vram)
    print(f"🧠 Total GPU memory: {total_gpu_memory}MB, usable: {usable_memory}MB")
    print(f"⚙️ Estimated max workers: {worker_count}")
    return int(worker_count)
def get_gpu(worker_count):
    return math.floor((1/worker_count)*100)/100

def record_temp(frames: list[bytes])-> str:
    print(f"Length:: {len(frames)}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp.close()
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bit PCM
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    return tmp.name