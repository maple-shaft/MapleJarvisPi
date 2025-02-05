from audio_input import MicrophoneAudioInput
from audio_output import AudioOutput
from audio_to_text import AudioRecorder
from pi_client import PiClient
import asyncio
import wave
import pyaudio as pa
import numpy as np
import threading as t
from scipy import signal

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
FORMAT = pa.paInt16
CHANNELS = 1
DEVICE_INDEX = 1
HOST = "10.0.0.169"
PORT = 9100

print("Starting MapleJarvisPi")

#audio_input = MicrophoneAudioInput(sample_rate=16000, format=pa.paInt16, channels=1, device_index=1)
#aur = AudioRecorder(audio_input=audio_input)
#client = PiClient(host=HOST, port = PORT)
#client_task = asyncio.create_task(client.start())

#print("main.py: PiClient started.")

#audio_output = AudioOutput(client=client,
#                           sample_rate=SAMPLE_RATE,
#                           sample_width=SAMPLE_WIDTH,
#                           channels=CHANNELS)
#audio_output_task = asyncio.create_task(audio_output.start())
#audio_output.start()

def thing(chunk):
    pcm_data = chunk.astype(np.float32) / 32768.0
    data_16000 = signal.resample_poly(pcm_data, 16000, 44100)
    return data_16000.astype(np.int16).tobytes()

def debug_write_to_file(audio16, filepath : str = "/tmp/pitest.wav"):
    try:
        with wave.open(filepath, "wb") as wv:
            wv.setnchannels(CHANNELS)
            wv.setframerate(SAMPLE_RATE)
            wv.setsampwidth(SAMPLE_WIDTH)
            wv.writeframes(audio16)
    except Exception as e:
        print(f"main:debug_write_to_file: Exception encountered: {e}")

async def main_audio_capture(aur : AudioRecorder, client : PiClient):
    print("Starting main_audio_capture coroutine...")
    try:
        while True:
            audio_data = aur.get_audio_for_transcription()
            audio16 = np.int16(audio_data * 32767)
            client.send(audio16)
            print(f"Audio data length: {len(audio16)}")
            #debug_write_to_file(audio16)
    except KeyboardInterrupt:
        print("Interrupt signal received, gracefully shutting down.")
    finally:
        aur.shutdown() # VERIFY
        audio_input.cleanup() # VERIFY
        client.stop() # VERIFY

async def main():
    print("Starting main coroutine...")
    audio_input = MicrophoneAudioInput(sample_rate=16000, format=pa.paInt16, channels=1, device_index=1)
    aur = AudioRecorder(audio_input=audio_input)
    client = PiClient(host=HOST, port = PORT)
    audio_output = AudioOutput(client=client,
                           sample_rate=SAMPLE_RATE,
                           sample_width=SAMPLE_WIDTH,
                           channels=CHANNELS)

    # Create asyncio tasks
    async with asyncio.TaskGroup() as tg:
        audio_capture_task = tg.create_task(main_audio_capture(aur, client))
        audio_output_task = tg.create_task(audio_output.start())
        client_task = tg.create_task(client.start())

if __name__ == "__main__":
    asyncio.run(main())

