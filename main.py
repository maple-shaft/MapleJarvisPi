from audio_input import MicrophoneAudioInput
from audio_output import AudioOutput
from audio_to_text import AudioRecorder
from pi_client import PiClient
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

audio_input = MicrophoneAudioInput(sample_rate=16000, format=pa.paInt16, channels=1, device_index=1)
aur = AudioRecorder(audio_input=audio_input)
client = PiClient(host=HOST, port = PORT)
client.start()

audio_output = AudioOutput(client=client,
                           sample_rate=SAMPLE_RATE,
                           sample_width=SAMPLE_WIDTH,
                           channels=CHANNELS)
audio_output.start()

def thing(chunk):
    pcm_data = chunk.astype(np.float32) / 32768.0
    data_16000 = signal.resample_poly(pcm_data, 16000, 44100)
    return data_16000.astype(np.int16).tobytes()

while True:
    audio_data = aur.get_audio_for_transcription()
    audio16 = np.int16(audio_data * 32767)
    client.send(audio16)
    print(f"Audio data length: {len(audio16)}")
    with wave.open("/tmp/pitest.wav", "wb") as wv:
        wv.setnchannels(1)
        wv.setframerate(16000)
        wv.setsampwidth(2)
        wv.writeframes(audio16)
