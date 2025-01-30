from audio_input import MicrophoneAudioInput
import time
import wave
import pyaudio
import numpy as np
from scipy import signal
#import matplotlib.pyplot as plt

audio_input = MicrophoneAudioInput(
    sample_rate=16000, format=pyaudio.paInt16, channels=1, device_index=1)

#audio_input.list_devices()

audio_input.setup()
print("Hello World")
