
import pyaudio as pa
import numpy as np
import pydub.playback
import scipy as sp
import time
import io
import sounddevice as sd
from threading import Thread, Event
import queue as q
from multiprocessing import Queue
from pydub import AudioSegment
from pydub.effects import speedup
from pydub.playback import play

class AudioOutput:

    def __init__(self,
         recv_queue : Queue,
         sample_rate, sample_width,
         channels,
         shutdown_event : Event,
         executor = None,
         debug = False):
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.recv_queue : Queue = recv_queue
        self.debug = debug
        if executor:
            self.executor = executor
        else:
            self.fetch_thread = Thread(target=self._fetch_and_play)
        self.shutdown_event : Event = shutdown_event,
        self.executor = executor

    def start(self):
        debug = self.debug
        print("Starting AudioOutput") if debug else None
        print(f"Devices: {sd.query_devices()}") if debug else None
        if self.executor:
            self.executor.submit(self._fetch_and_play)
        else:
            self.fetch_thread.start()

    def _play(self, data : np.ndarray, sample_rate : int, speedup_flag : bool = True):
        buffer = io.BytesIO(data.tobytes())
        aus = AudioSegment.from_raw(buffer, sample_width=2, frame_rate=sample_rate, channels=1)
        if speedup_flag:
            aus = speedup(aus, 1.25)
        play(aus)

    def _fetch_and_play(self):
        debug = self.debug
        print(f"AudioOutput: Starting fetch_and_play runnable {self.shutdown_event}") if debug else None
        while True:
            try:
                data = self.recv_queue.get(timeout = 2)
                print(f"AudioOutput.fetch_and_play: Returned and data is {type(data)}") if debug else None
                if data is None:
                    time.sleep(1)
                    continue
                else:
                    print(f"AudioOutput.fetch_and_play: Next bytes fetched for playing: {type(data)}") if debug else None
                    print(f"AudioOutput.fetch_and_play: Devices: {sd.query_devices()}") if debug else None
                    
                    sd.play(data = data,
                            samplerate = 21000,
                            blocking = True
                    )
            except KeyboardInterrupt:
                print("AudioOutput: Captured KeyboardInterrupt!")
                self.shutdown_event.set()
            except q.Empty:
                pass
            except Exception as e:
                print(f"AudioOutput: Exception encountered in fetch_and_play {e}")
                break
