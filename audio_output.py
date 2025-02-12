
import pyaudio as pa
import numpy as np
import scipy as sp
import time
import sounddevice as sd
from pi_client import PiClient
from threading import Thread, Event
import queue as q
from multiprocessing import Queue

class AudioOutput:

    def __init__(self,
         recv_queue : Queue,
         sample_rate, sample_width,
         channels,
         shutdown_event : Event,
         executor = None):
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.recv_queue : Queue = recv_queue
        if executor:
            self.executor = executor
        else:
            self.fetch_thread = Thread(target=self._fetch_and_play)
        self.shutdown_event : Event = shutdown_event,
        self.executor = executor

    def start(self):
        print("Starting AudioOutput")
        print(f"Devices: {sd.query_devices()}")
        if self.executor:
            self.executor.submit(self._fetch_and_play)
        else:
            self.fetch_thread.start()

    def _fetch_and_play(self):
        print(f"AudioOutput: Starting fetch_and_play runnable {self.shutdown_event}")
        while True:
        #while not self.shutdown_event.is_set():
            #print("FETCH AND PLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYYYYYYYYYYYYY")
            try:
                #time.sleep(1)
                #print("We are about to check self.recv_queue!!!!!!ngonrgn4jt42jm24f")
                #time.sleep(3)
                data = self.recv_queue.get(timeout = 2)
                #print(f"SWWEEET, returned and data is {type(data)}")
                if data is None:
                    time.sleep(1)
                    continue
                else:
                    #print(f"Next bytes fetched for playing: {type(data)}")
                    #print(f"Devices: {sd.query_devices()}")
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
