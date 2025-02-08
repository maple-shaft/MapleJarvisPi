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
            client.send(audio16)
            print(f"Audio data length: {len(audio16)}")
            #debug_write_to_file(audio16)
    except KeyboardInterrupt:
        print("Interrupt signal received, gracefully shutting down.")
    finally:
        aur.shutdown() # VERIFY
        #audio_input.cleanup() # VERIFY
        #client.stop() # VERIFY

def callback(arg : str):
    print(f"Callback: {arg}")

def on_recording_start():
    callback("on_recording_start")

def on_recording_stop():
    callback("on_recording_stop")

def on_recorded_chunk(arg):
    callback("on_recorded_chunk")

def on_wakeword_detected():
    callback("on_wakeword_detected")

def on_wakeword_timeout():
    callback("on_wakeword_timeout")

def on_wakeword_detection_start():
    callback("on_wakeword_detection_start")

def on_wakeword_detection_end():
    callback("on_wakeword_detection_end")

if __name__ == "__main__":
    print("Starting main...")
    client = PiClient(host=HOST, port = PORT)
    client.start()

    main_audio_capture(client = client)

    #audio_input = MicrophoneAudioInput(sample_rate=16000, format=pa.paInt16, channels=1, device_index=1)

    #aur = AudioRecorder(audio_input=audio_input,
    #                    silero_vad_model = silero_vad_model)
                        #on_recording_start = on_recording_start,
                        #on_recording_stop = on_recording_stop,
                        #on_recorded_chunk = on_recorded_chunk,
                        #on_wakeword_detected = on_wakeword_detected,
                        #on_wakeword_timeout = on_wakeword_timeout,
                        #on_wakeword_detection_start = on_wakeword_detection_start,
                        #on_wakeword_detection_end = on_wakeword_detection_end)

    #audio_output = AudioOutput(client=client,
    #                       sample_rate=SAMPLE_RATE,
    #                       sample_width=SAMPLE_WIDTH,
    #                       channels=CHANNELS)
    #process_list = []
    #process_list.append(Process(target=main_audio_capture, args=[client]))
    #process_list.append(Process(target=main_audio_output, args=[audio_output]))
    #for pr in process_list:
        #pr.daemon = True
    #    pr.start()
        #pr.join()
    #    time.sleep(4)

    # Create asyncio tasks
    #async with asyncio.TaskGroup() as tg:
    #    audio_capture_task = tg.create_task(main_audio_capture(aur, client))
        #audio_output_task = tg.create_task(audio_output.start())
        #client_task = tg.create_task(client.start())

