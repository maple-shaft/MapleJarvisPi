from audio_input import MicrophoneAudioInput
from audio_to_text import AudioRecorder
import time
import wave
import pyaudio
import numpy as np
from scipy import signal
#import matplotlib.pyplot as plt

audio_input = MicrophoneAudioInput(sample_rate=16000, format=pyaudio.paInt16, channels=1, device_index=1)
aur = AudioRecorder(audio_input=audio_input)

print("Hello World")

def thing(chunk):
    pcm_data = chunk.astype(np.float32) / 32768.0
    #pcm_data = np.frombuffer(chunk, dtype=np.float32)
    #pcm_data = np.frombuffer(chunk, dtype=np.int16)
    data_16000 = signal.resample_poly(pcm_data, 16000, 44100)
    return data_16000.astype(np.int16).tobytes()

while True:
    #time.sleep(3)
    audio_data = aur.get_audio_for_transcription()
    audio16 = np.int16(audio_data * 32767)
    #plt.plot(audio16)
    #plt.show()
    print(f"Audio data length: {len(audio16)}")
    with wave.open("/tmp/pitest.wav", "wb") as wv:
        wv.setnchannels(1)
        wv.setframerate(16000)
        wv.setsampwidth(2)
        wv.writeframes(audio16)
