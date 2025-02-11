
import pyaudio as pa
import numpy as np
import scipy as sp
import time
import sounddevice as sd
from pi_client import PiClient
from threading import Thread
import queue as q
from multiprocessing import Queue

class AudioOutput:

    def __init__(self,
         recv_queue : Queue,
         sample_rate, sample_width,
         channels,
         shutdown_event,
         executor = None):
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.recv_queue = recv_queue
        if executor:
            self.executor = executor
        else:
            self.fetch_thread = Thread(target=self._fetch_and_play)
        self.shutdown_event = shutdown_event,
        self.executor = executor

    def start(self):
        print("Starting AudioOutput")
        print(f"Devices: {sd.query_devices()}")
        if self.executor:
            self.executor.submit(self._fetch_and_play)
        else:
            self.fetch_thread.start()

    def _fetch_and_play(self):
        print("AudioOutput: Starting fetch_and_play runnable")
        while not self.shutdown_event.is_set():
            try:
                data = self.recv_queue.get(timeout=0.3)
                if data is None:
                    time.sleep(1)
                    continue
                else:
                    print(f"Next bytes fetched for playing: {type(data)}")
                    print(f"Devices: {sd.query_devices()}")
                    sd.play(data = data,
                            samplerate = self.sample_rate,
                            blocking = True
                    )
            except KeyboardInterrupt:
                print("AudioOutput: Captured KeyboardInterrupt!")
                self.shutdown_event.set()
            except q.Empty:
                pass
            except Exception as e:
                print(f"AudioOutput: Exception encountered in fetch_and_play {e}")
