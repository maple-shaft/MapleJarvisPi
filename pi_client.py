import numpy as np
from audio_input import MicrophoneAudioInput
from audio_to_text import AudioRecorder
import pyaudio as pa
import socket
import pickle
import queue as q
import asyncio
from asyncio.exceptions import CancelledError
from typing import Any, AsyncGenerator

class PiClient:

    def __init__(self,
                 host : str,
                 port = int):
        self.host = host
        self.port = port
        self.recv_queue = q.SimpleQueue()

    async def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.connect((self.host, self.port))
        print(f"Server connecting to {self.host}:{self.port}")
        # start receive loop
        await self.receive()

    def send(self, obj):
        val = pickle.dumps(obj)
        length = len(val)
        length = str(length).rjust(8,"0")
        self.socket.sendall(bytes(length, "utf-8"))
        self.socket.sendall(val)

    async def receive(self):
        print("PiClient.receive: Starting receive")
        try:
            while True:
                try:
                    length = self.socket.recv(8)
                    length = int(length.decode("utf-8"))
                    full_length = length
                    message = None
                    while length > 0:
                        chunk_len = min(128, length)
                        length -= chunk_len
                        chunk = self.socket.recv(chunk_len)
                        if message is None:
                            message = chunk
                        else:
                            message = message + chunk
                    while len(message) < full_length:
                        chunk_len = min(128, full_length - len(message))
                        chunk = self.socket.recv(chunk_len)
                        message = message + chunk

                    # now that we have everything, pickle it to an object
                    obj = pickle.loads(message)
                    print(f"PiClient.receive: Received obj {obj} of type {type(obj)}")
                    self.recv_queue.put(obj)
                except Exception as e:
                    print(f"PiClient.receive: Exception encountered in receive: {e}")
                    break
        except CancelledError:
            print("PiClient.receive: Interrupt signal received, gracefully shutting down...")
            self.finalize()
    
    def finalize(self):
        if self.socket:
            self.socket.close()

    def next_bytes(self):
        #print("Starting next bytes")
        if not self.recv_queue.empty():
            return self.recv_queue.get()
        else:
            return None
    
