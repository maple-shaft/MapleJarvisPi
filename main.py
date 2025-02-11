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
from multiprocessing import Process, get_context, Event, Queue
from scipy import signal
import concurrent.futures

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

def main_audio_capture(send_queue : Queue, shutdown_event : Event, executor):
    print("Starting main_audio_capture coroutine...")
    # Setup voice activity detection model Silero VAD
    #try:
    #    silero_vad_model, _ = torch.hub.load(
    #        repo_or_dir="snakers4/silero-vad",
    #        model="silero_vad",
    #        verbose=False,
    #        onnx=False
    #    )
    #except Exception as e:
    #    print(f"AudioRecorder.__init__: Error initializing Silero VAD")

    audio_input = MicrophoneAudioInput(
         sample_rate=16000,
         format=pa.paInt16,
         channels=1,
         device_index=1)
    aur = AudioRecorder(audio_input=audio_input,
                       # silero_vad_model = silero_vad_model,
                        shutdown_event = shutdown_event,
                        executor = executor,
                        debug = False)
    try:
        while not shutdown_event.is_set():
            try:
                audio_data = aur.wait_audio()
                time.sleep(0.1)
                print("main.py: GOT AUDIO_DATA!!!")
                audio16 = np.int16(audio_data * 32767)
                print("main.py: about to send audio data to client.")
                send_queue.put(audio16)
                print(f"main.py: Audio data length: {len(audio16)}")
            except KeyboardInterrupt:
                print("Interrupt signal received, gracefully shutting down.")
                shutdown_event.set()
    finally:
        aur.shutdown()

if __name__ == "__main__":
    print("Starting main...")
    shutdown_event = Event()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        client = PiClient(host=HOST,
                          port = PORT,
                          shutdown_event = shutdown_event,
                          debug = False,
                          executor = executor)
        client.start()
        audio_output = AudioOutput(
             recv_queue = client.recv_queue,
             sample_rate = 21000,
             sample_width = SAMPLE_WIDTH,
             channels = 1,
             shutdown_event = shutdown_event,
             executor = executor)
        audio_output.start()
        time.sleep(2.0)
        main_audio_capture(send_queue = client.send_queue, shutdown_event = shutdown_event, executor = executor)

