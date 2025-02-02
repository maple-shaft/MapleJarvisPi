import pyaudio as pa
import numpy as np
import threading as t
import scipy as sp
import time
import sounddevice as sd
from pi_client import PiClient

class AudioOutput:

    def __init__(self, client : PiClient, sample_rate, sample_width, channels):
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.thread = t.Thread(target=self._fetch_and_play, name="audio_output_thread")
        self.client = client

    def start(self):
        print("Starting AudioOutput")
        self.thread.run()

    def _fetch_and_play(self):
        print("Starting fetch_and_play runnable")
        while True:
            try:
                nb = self.client.next_bytes()
                if not nb:
                    time.sleep(1)
                    continue
                else:
                    print(f"Next bytes fetched for playing: {type(nb)}")
                    sd.play(data = nb,
                            sample_rate = self.sample_rate,
                            blocking = True
                    )
            except Exception as e:
                print(f"Exception encountered in fetch_and_play {e}")

                