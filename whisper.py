import numpy as np
import librosa
import sys
import soundfile as sf
from functools import lru_cache
from common.constant import WHISPER_LANG_CODES,SAMPLING_RATE
from typing import Any, List, Tuple, Optional

import time
from pkg.logger.logger import setup_logger
import io


# Setup logger
logger = setup_logger(__name__)

@lru_cache(10**6)
def load_audio(file_name: str) -> np.ndarray:
    a, _ = librosa.load(file_name,sr=SAMPLING_RATE,dtype=np.float32)
    return a

def load_audio_chunk(file_name:str,beg:float,end:float) -> np.ndarray:
    audio = load_audio(file_name)
    begin_size = (beg*SAMPLING_RATE)
    end_size = (end*SAMPLING_RATE)
    return audio[begin_size:end_size]

class ASRBase:
    sep = " "
    # join transcribe words with this character (" " for whisper_timestamped,
    # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self,language: str,model_size: Optional[str] = None,cache_dir: Optional[str] = None,model_dir: Optional[str] = None):
        self.transcribe_kargs = {}
        self.original_language = None if language == "auto" else language
        self.model = self.load_model(model_size,cache_dir,model_dir)

    def load_model(self,model_size,cache_dir,model_dir):
        raise NotImplemented("must be implemented in the child class")
    
    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")

class FasterWhisperASR(ASRBase):
    """
        Uses faster-whisper library as the backend. 
        Works much faster, appx 4-times (in offline mode). 
        For GPU, it requires installation with a specific CUDNN version.
    """

    sep = ""

    def load_model(self, model_size=None, cache_dir=None,model_dir=None):
        from faster_whisper import WhisperModel,BatchedInferencePipeline
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. model_size and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("model_size or model_dir parameter must be set")
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel(model_size_or_path,device="cuda",compute_type="float16",download_root=cache_dir)
        # model = WhisperModel(model_size_or_path,download_root=cache_dir)
        model = BatchedInferencePipeline(model=model)
        return model
        
    def transcribe(self, audio, init_prompt=""):
        segments , _info = self.model.transcribe(audio,
                                                language=self.original_language,
                                                initial_prompt=init_prompt,
                                                beam_size=5,
                                                word_timestamps=True,
                                                condition_on_previous_text=True,
                                                **self.transcribe_kargs)
        return list(segments)
    
    def transcribe_bytes(self,bytes_data: bytes,language: str) -> Tuple[str,str]:
        audio_file = io.BytesIO(bytes_data)
        start_request_time = time.perf_counter()
        segments, _info = self.model.transcribe(audio_file,
                                                beam_size=5,
                                                temperature=0,
                                                language=language,
                                                word_timestamps=True,
                                                # condition_on_previous_text=True,
                                                **self.transcribe_kargs)
        full_text = "".join(segment.text for segment in segments)
        request_time = round(time.perf_counter() - start_request_time, 4)
        return full_text.strip(),f"{request_time}s"
    
    def ts_words(self, segments):
        results = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                results.append(t)
        return results

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"

    
class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper library as the backend, optimized for Apple Silicon.
    Models available: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
    Significantly faster than faster-whisper (without CUDA) on Apple M1. 
    """
    sep = " "
    def load_model(self, model_size, cache_dir, model_dir):
        """
            Loads the MLX-compatible Whisper model.

            Args:
                modelsize (str, optional): The size or name of the Whisper model to load. 
                    If provided, it will be translated to an MLX-compatible model path using the `translate_model_name` method.
                    Example: "large-v3-turbo" -> "mlx-community/whisper-large-v3-turbo".
                cache_dir (str, optional): Path to the directory for caching models. 
                    **Note**: This is not supported by MLX Whisper and will be ignored.
                model_dir (str, optional): Direct path to a custom model directory. 
                    If specified, it overrides the `model_size` parameter.
        """
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx # Is installed with mlx-whisper
        
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used.")
            model_size_or_path = model_dir
        elif model_size is not None:
            model_size_or_path = self.translate_model_name(model_size)
            logger.debug(f"Loading whisper model {model_size}. You use mlx whisper, so {model_size_or_path} will be used.")
        
        self.model_size_or_path = model_size_or_path
        
        # Note: ModelHolder.get_model loads the model into a static class variable, 
        # making it a global resource. This means:
        # - Only one model can be loaded at a time; switching models requires reloading.
        # - This approach may not be suitable for scenarios requiring multiple models simultaneously,
        #   such as using whisper-streaming as a module with varying model sizes.
        dtype = mx.float16 # Default to mx.float16. In mlx_whisper.transcribe: dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
        ModelHolder.get_model(model_size_or_path, dtype) #Model is preloaded to avoid reloading during transcription
        
        return transcribe
    def translate_model_name(self, model_name):
        """
        Translates a given model name to its corresponding MLX-compatible model path.

        Args:
            model_name (str): The name of the model to translate.

        Returns:
            str: The MLX-compatible model path.
        """
        # Dictionary mapping model names to MLX-compatible paths
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx"
        }
        # Retrieve the corresponding MLX model path
        mlx_model_path = model_mapping.get(model_name)

        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")
    def transcribe(self, audio, init_prompt=""):
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
            **self.transcribe_kargs
        )
        return segments.get("segments", [])


    def ts_words(self, segments):
        """
        Extract timestamped words from transcription segments and skips words with high no-speech probability.
        """
        return [
            (word["start"], word["end"], word["word"])
            for segment in segments
            for word in segment.get("words", [])
            if segment.get("no_speech_prob", 0) <= 0.9
        ]
    
    def segments_end_ts(self, res):
        return [s['end'] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"

class HypothesisBuffer:
    
    def __init__(self):
        self.committed_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_committed_time = 0
        self.last_committed_word = None
    
    def insert(self,new,offset):
        # compare self.committed_in_buffer and new. It inserts only the words in new that extend the committed_in_buffer, 
        # it means they are roughly behind last_committed_time and new in content
        # the new tail is added to self.new

        new = [(a+offset,b+offset,t) for a,b,t in new]

        self.new = [(a,b,t) for a,b,t in new if a > self.last_committed_time - 0.1]

        if len(self.new) >= 1:
            a,b,t = self.new[0]

            if abs(a - self.last_committed_time) < 1:
                if self.committed_in_buffer:
                # it's going to search for 1, 2, ..., 5 
                # consecutive words (n-grams) that are identical in committed and new. If they are, they're dropped.
                    cn = len(self.committed_in_buffer)
                    nn = len(self.new)

                    for i in range(1,min(min(cn,nn),5)+1): #5 is the maximum
                        c = " ".join([self.committed_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                        
                        if c == tail:
                            words = []
                            for _ in range(i):
                                words.append(repr(self.new.pop(0)))
                            
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break
    
    def flush(self):
        # returns committed chunk = the longest common prefix of 2 last inserts.
        commit = []
        while self.new:
            na,nb,nt = self.new[0]
            if len(self.buffer) == 0:
                break
            if nt == self.buffer[0][2]:
                commit.append((na,nb,nt))
                self.last_committed_time = nb
                self.last_committed_word = nt

                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit

    def pop_committed(self,time):
        while self.committed_in_buffer and self.committed_in_buffer[0][1] <= time:
            self.committed_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    SAMPLING_RATE = 16000
    def __init__(self,asr,tokenizer=None,buffer_trimming=("segment",15)):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.init()
        self.buffer_trimming_way,self.buffer_trimming_sec = buffer_trimming
    
    def init(self,offset=None):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer()
        self.buffer_time_offset = 0

        if offset is not None:
            self.buffer_time_offset = offset
        
        self.transcript_buffer.last_committed_time = self.buffer_time_offset

        self.committed = []

    def insert_audio_chunk(self,audio):
        self.audio_buffer = np.append(self.audio_buffer,audio)

    def prompt(self):
        """
        Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of committed text that is inside of the scrolled away part of audio buffer. 
        "context" is the committed text that is inside the audio buffer. 
        It is transcribed again and skipped. 
        It is returned only for debugging and logging reasons.
        """
        k = max(0,len(self.committed)-1)
        while k > 0 and self.committed[k-1][1] > self.buffer_time_offset:
            k -= 1

        p = self.committed[:k]
        p = [t for _,_,t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.committed[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t in non_prompt)
        
    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
        The non-emty text is confirmed (committed) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)
        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.committed.extend(o)
        completed = self.to_flush(o)
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        # there is a newly confirmed text

        if o and self.buffer_trimming_way == "sentence":  # trim the completed sentences
            if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:  # longer than this
                self.chunk_completed_sentence()

        
        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30 # if the audio buffer is longer than 30s, trim it
        
        if len(self.audio_buffer)/self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

            # alternative: on any word
            #l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # let's find committed word that is less
            #k = len(self.committed)-1
            #while k>0 and self.committed[k][1] > l:
            #    k -= 1
            #t = self.committed[k][1] 
            logger.debug("chunking segment")
            #self.chunk_at(t)

        logger.debug(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}")
        return self.to_flush(o)
    
    def chunk_completed_sentence(self):
        if self.committed == []: return
        logger.debug(self.committed)
        sends = self.words_to_sentences(self.committed)
        for s in sends:
            logger.debug(f"\t\tSENT: {s}")
        if len(sends) < 2:
            return
        while len(sends) > 2:
            sends.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sends[-2][1]

        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.committed == []: return

        ends = self.asr.segments_end_ts(res)

        t = self.committed[-1][1]

        if len(ends) > 1:

            e = ends[-2]+self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2]+self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within committed area")
        else:
            logger.debug(f"--- not enough segments to chunk")
    
    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time"
        """
        self.transcript_buffer.pop_committed(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """
        Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """
        
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            f_sent = sent
            while cwords:
                b,e,w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg,end,f_sent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, non_committed: {f}")
        self.buffer_time_offset += len(self.audio_buffer)/16000
        return f


    def to_flush(self, sends, sep=None, offset=0, ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sends: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sends)
        if len(sends) == 0:
            b = None
            e = None
        else:
            b = offset + sends[0][0]
            e = offset + sends[-1][1]
        return (b,e,t)
    
class VACOnlineASRProcessor(OnlineASRProcessor):
    """
    Wraps OnlineASRProcessor with VAC (Voice Activity Controller). 
    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds), 
    it runs VAD and continuously detects whether there is speech or not. 
    When it detects end of speech (non-voice for 500ms), it makes OnlineASRProcessor to end the utterance immediately.
    """

    def __init__(self, online_chunk_size, *a, **kw):
        self.online_chunk_size = online_chunk_size
        self.online = OnlineASRProcessor(*a, **kw)

        # VAC:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        from silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(model)  # we use the default options there: 500ms silence, 100ms padding, etc.  

        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0

        self.is_currently_final = False

        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([],dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([],dtype=np.float32)


    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0]-self.buffer_offset
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"]-self.buffer_offset
                end = res["end"]-self.buffer_offset
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # We keep 1 second because VAD may later find start of voice in it.
                # But we trim it to prevent OOM. 
                self.buffer_offset += max(0,len(self.audio_buffer)-self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]


    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE*self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            print("no online update, only VAD", self.status)
            return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret
    

def create_tokenizer(language):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert language in WHISPER_LANG_CODES, "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if language == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if language in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from mosestokenizer import MosesTokenizer
        return MosesTokenizer(language)

    # the following languages are in Whisper, but not in wtpsplit:
    if language in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        logger.debug(f"{language} code is not supported by wtpsplit. Going to use None lang_code option.")
        language = None

    from wtpsplit import WtP
    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=language)
    return WtPtok()



def add_shared_args(parser):
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0, help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
    parser.add_argument('--model', type=str, default='large-v3-turbo', choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(","),help="Name size of the Whisper model to use (default: large-v3). The model is automatically downloaded from the model hub if not present in model cache dir.")
    parser.add_argument('--model_cache_dir', type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved")
    parser.add_argument('--model_dir', type=str, default=None, help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    parser.add_argument('--language', '--lan', type=str, default='auto', help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="faster-whisper", choices=["faster-whisper", "whisper_timestamped", "mlx-whisper"],help='Load only this backend for Whisper processing.')
    parser.add_argument('--vac', action="store_true", default=False, help='Use VAC = voice activity controller. Recommended. Requires torch.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False, help='Use VAD = voice activity detection, with the default parameters.')
    parser.add_argument('--buffer_trimming', type=str, default="segment", choices=["sentence", "segment"],help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15, help='Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.')

def asr_factory(args):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    backend = args.backend
    if backend == "faster-whisper":
        asr_cls = FasterWhisperASR
    else:
        asr_cls = MLXWhisper
    # Only for FasterWhisperASR and WhisperTimestampedASR
    size = args.model
    t = time.time()
    #Setup log
    # logger = setup_logger(__name__)
    logger.info(f"Loading Whisper {size} model for {args.language}...")
    asr = asr_cls(model_size=size, language=args.language, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
    e = time.time()
    logger.info(f"done. It took {round(e-t,2)} seconds.")
    
    # Apply common configurations
    if getattr(args, 'vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.language
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    # Create the OnlineASRProcessor
    if args.vac:
        online = VACOnlineASRProcessor(args.min_chunk_size, asr,tokenizer,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    else:
        online = OnlineASRProcessor(asr,tokenizer,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))

    return asr, online

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')
    
    args = parser.parse_args()

    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

    #setup log
    # setup_logger(__name__)


    audio_path = args.audio_path

    duration = len(load_audio(audio_path))/SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path,0,1)

    # warm up the ASR because the very first transcribe takes much more time than the other
    asr.transcribe(a)

    beg = args.start_at
    start = time.time()-beg

    def output_transcript(o, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time()-start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
        else:
            # No text, so no output
            pass

    if args.offline: ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode 
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path,beg,end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.debug(f"## last processed {end:.2f}s")

            if end >= duration:
                break
            
            beg = end
            
            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else: # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < end+min_chunk:
                time.sleep(min_chunk+end-now)
            end = time.time() - start
            a = load_audio_chunk(audio_path,beg,end)
            beg = end
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")

            if end >= duration:
                break
        now = None

    o = online.finish()
    output_transcript(o, now=now)