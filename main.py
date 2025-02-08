from audio_input import MicrophoneAudioInput
from audio_output import AudioOutput
from audio_to_text import AudioRecorder
from pi_client import PiClient
import asyncio
import wave
import pyaudio as pa
import numpy as np
import torch
import threading as t
import time
from multiprocessing import Process, get_context
from scipy import signal

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
FORMAT = pa.paInt16
CHANNELS = 1
DEVICE_INDEX = 1
HOST = "10.0.0.169"
PORT = 9100

print("Starting MapleJarvisPi")

def debug_write_to_file(audio16, filepath : str = "/tmp/pitest.wav"):
    try:
        with wave.open(filepath, "wb") as wv:
            wv.setnchannels(CHANNELS)
            wv.setframerate(SAMPLE_RATE)
            wv.setsampwidth(SAMPLE_WIDTH)
            wv.writeframes(audio16)
    except Exception as e:
        print(f"main:debug_write_to_file: Exception encountered: {e}")

def main_audio_output(audio_output):
    print("Starting main_audio_output")
    audio_output.start()

def main_audio_capture(client : PiClient):
    print("Starting main_audio_capture coroutine...")
    # Setup voice activity detection model Silero VAD
    try:
        silero_vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            verbose=False,
            onnx=False
        )
    except Exception as e:
        print(f"AudioRecorder.__init__: Error initializing Silero VAD")

    audio_input = MicrophoneAudioInput(sample_rate=16000, format=pa.paInt16, channels=1, device_index=1)
    aur = AudioRecorder(audio_input=audio_input,
                        silero_vad_model = silero_vad_model)
    try:
        while True:
            audio_data = aur.wait_audio()
            print("main.py: GOT AUDIO_DATA!!!")
            audio16 = np.int16(audio_data * 32767)
            print("main.py: about to send audio data to client.")
            client.send(audio16)
            print(f"main.py: Audio data length: {len(audio16)}")
            #break
            #debug_write_to_file(audio16)
    except KeyboardInterrupt:
        print("Interrupt signal received, gracefully shutting down.")
    finally:
        aur.shutdown() # VERIFY
        #audio_input.cleanup() # VERIFY
        #client.stop() # VERIFY

if __name__ == "__main__":
    print("Starting main...")
    client = PiClient(host=HOST, port = PORT)
    client.start()
    audio_output = AudioOutput(
         recv_queue = client.recv_queue,
         sample_rate = SAMPLE_RATE,
         sample_width = SAMPLE_WIDTH,
         channels = 1)
    main_audio_capture(client = client)
    time.sleep(4)
    audio_output.start()


