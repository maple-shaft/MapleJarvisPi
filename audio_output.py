
import pyaudio as pa
import numpy as np
import scipy as sp
import time
import sounddevice as sd
from pi_client import PiClient
from threading import Thread
from multiprocessing import Queue

class AudioOutput:

    def __init__(self, recv_queue : Queue, sample_rate, sample_width, channels):
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.recv_queue = recv_queue
        self.fetch_thread = Thread(target=self._fetch_and_play)

    def start(self):
        print("Starting AudioOutput")
        self.fetch_thread.start()

    def next_bytes(self):
        #print("Starting next bytes")
        if not self.recv_queue.empty():
            return self.recv_queue.get()
        else:
            return None

    def _fetch_and_play(self):
        print("Starting fetch_and_play runnable")
        while True:
            try:
                data = self.next_bytes()
                if data == None:
                    time.sleep(1)
                    continue
                else:
                    print(f"Next bytes fetched for playing: {type(nb)}")
                    sd.play(data = data,
                            sample_rate = self.sample_rate,
                            blocking = True
                    )
            except KeyboardInterrupt:
                print("AudioOutput: Captured KeyboardInterrupt!")
                break
            except Exception as e:
                print(f"Exception encountered in fetch_and_play {e}")
                
