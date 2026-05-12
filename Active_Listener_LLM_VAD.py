"""
VAD-Based Active Listening Pipeline with LLM + Decibel Barge-In
Author: Hamza Ahmed Durrani
"""

import sounddevice as sd
import webrtcvad
import whisper
import numpy as np
import collections
import queue
import threading
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, sys

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")


SAMPLE_RATE            = 16000
FRAME_DURATION_MS      = 30
FRAME_SIZE             = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
CHANNELS               = 1

VAD_AGGRESSIVENESS     = 2
SPEECH_PADDING_MS      = 300
SPEECH_TRIGGER_FRAMES  = 3
SILENCE_TRIGGER_FRAMES = 15
MIN_SPEECH_SEC         = 0.5

# RMS energy of the mic signal converted to dBFS (0 = full scale, -60 = near silence).
# If the user's voice exceeds this level WHILE the bot is speaking, TTS is killed.
# Raise if background noise causes false triggers.
# Lower if you have to speak very loudly to interrupt.
BARGE_IN_DB_THRESHOLD  = -23

# How many consecutive loud frames before we accept it as intentional speech
# (prevents a single click or breath from interrupting)
BARGE_IN_CONFIRM_FRAMES = 3

WHISPER_MODEL_SIZE = "base"
LLM_MODEL_NAME     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_PROMPT = """
You are a concise cooking assistant robot. Answer in 1-2 short sentences only.
Never use bullet points or lists. Speak naturally as if talking out loud.
"""


def rms_dbfs(audio_int16_bytes: bytes) -> float:
    """Return RMS level of a frame in dBFS. Returns -96 for silence."""
    samples = np.frombuffer(audio_int16_bytes, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-6:
        return -96.0
    return 20.0 * np.log10(rms / 32768.0)


class BargeInController:
    """
    Background thread always reads the mic.

    While bot is NOT speaking  → normal VAD-based speech capture.
    While bot IS speaking      → dB threshold check only.
                                 If BARGE_IN_CONFIRM_FRAMES consecutive frames
                                 exceed BARGE_IN_DB_THRESHOLD, TTS is killed
                                 and we switch to full VAD capture mode.
    """

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.audio_queue = queue.Queue()

        self.tts_process  = None
        self._tts_lock    = threading.Lock()
        self._bot_talking = False

        self._frames      = []
        self._ready       = threading.Event()
        self._capturing   = False
        self._sil_count   = 0
        self._loud_count  = 0          # consecutive frames above dB threshold

        padding_frames = int(SPEECH_PADDING_MS / FRAME_DURATION_MS)
        self._ring = collections.deque(maxlen=padding_frames)

        t = threading.Thread(target=self._vad_loop, daemon=True)
        t.start()

    def _audio_callback(self, indata, frames, time, status):
        audio_int16 = (indata[:, 0] * 32768).astype(np.int16)
        self.audio_queue.put(audio_int16.tobytes())

    def _vad_loop(self):
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            dtype='float32', blocksize=FRAME_SIZE,
            callback=self._audio_callback
        ):
            while True:
                frame_bytes = self.audio_queue.get()

                # While bot is talking: only do dB threshold check 
                if self._bot_talking:
                    db = rms_dbfs(frame_bytes)
                    if db > BARGE_IN_DB_THRESHOLD:
                        self._loud_count += 1
                        if self._loud_count >= BARGE_IN_CONFIRM_FRAMES:
                            self._loud_count = 0
                            # Kill TTS
                            with self._tts_lock:
                                if self.tts_process and self.tts_process.poll() is None:
                                    self._bot_talking = False
                                    self.tts_process.kill()
                                    self.tts_process = None
                                    print(f"\r🛑 Interrupted (dB: {db:.1f})      ")
                            # Seed the ring buffer with this loud frame and
                            # immediately start capturing
                            self._capturing   = True
                            self._sil_count   = 0
                            self._frames      = [frame_bytes]
                            self._ring.clear()
                            self._ready.clear()
                    else:
                        self._loud_count = 0
                    continue

                try:
                    is_speech = self.vad.is_speech(frame_bytes, SAMPLE_RATE)
                except Exception:
                    is_speech = False

                if not self._capturing:
                    self._ring.append((frame_bytes, is_speech))
                    if sum(1 for _, s in self._ring if s) > SPEECH_TRIGGER_FRAMES:
                        self._capturing = True
                        self._sil_count = 0
                        self._frames    = [f for f, _ in self._ring]
                        self._ring.clear()
                        self._ready.clear()
                else:
                    self._frames.append(frame_bytes)
                    self._sil_count = 0 if is_speech else self._sil_count + 1
                    if self._sil_count > SILENCE_TRIGGER_FRAMES:
                        self._capturing = False
                        self._ready.set()

    def wait_for_speech(self):
        """Block until a full utterance is captured. Returns float32 array."""
        while True:
            print("\r🎤 Listening...          ", end="", flush=True)
            self._ready.wait()
            self._ready.clear()

            audio_np  = np.frombuffer(b"".join(self._frames), dtype=np.int16).astype(np.float32)
            audio_np /= 32768.0

            if len(audio_np) / SAMPLE_RATE >= MIN_SPEECH_SEC:
                return audio_np

    def speak(self, text: str):
        """Speak via macOS `say`. dB monitor is active for the duration."""
        print(f"\r🔊 Bot: {text}")
        with self._tts_lock:
            self._bot_talking = True
            self._loud_count  = 0
            proc = subprocess.Popen(
                ["say", "-r", "175", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.tts_process = proc

        proc.wait()

        with self._tts_lock:
            self._bot_talking = False
            self._loud_count  = 0
            if self.tts_process is proc:
                self.tts_process = None


class WhisperTranscriber:
    def __init__(self):
        sys.stdout = open(os.devnull, 'w')
        self.model = whisper.load_model(WHISPER_MODEL_SIZE, device="cpu")
        sys.stdout = sys.__stdout__

    def transcribe(self, audio: np.ndarray) -> str:
        result = self.model.transcribe(audio, fp16=False)
        text   = result["text"].strip()
        return "" if self._hallucination(text) else text

    @staticmethod
    def _hallucination(text: str) -> bool:
        words = text.split()
        if not words:
            return True
        return max(words.count(w) for w in set(words)) / len(words) > 0.5


class LLMCommandRouter:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto"
        )
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def respond(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})

        dev  = next(self.model.parameters()).device
        tok  = self.tokenizer.apply_chat_template(
                   self.history, return_tensors="pt",
                   add_generation_prompt=True, tokenize=True)
        ids  = (tok if isinstance(tok, torch.Tensor) else tok["input_ids"]).to(dev)
        mask = torch.ones_like(ids)
        plen = ids.shape[-1]

        with torch.no_grad():
            out = self.model.generate(
                ids, attention_mask=mask,
                max_new_tokens=80,
                temperature=0.7, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
        self.history.append({"role": "assistant", "content": response})
        return response

def run_pipeline():
    print("CookBot ready. Speak at any time.")
    print(f"Barge-in threshold: {BARGE_IN_DB_THRESHOLD} dBFS  "
          f"(raise if background noise interrupts, lower if hard to interrupt)\n")

    barge       = BargeInController()
    transcriber = WhisperTranscriber()
    llm         = LLMCommandRouter()

    try:
        while True:
            audio = barge.wait_for_speech()
            text  = transcriber.transcribe(audio)
            if not text:
                continue
            print(f"\r👤 You: {text}")
            response = llm.respond(text)
            barge.speak(response)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    run_pipeline()
