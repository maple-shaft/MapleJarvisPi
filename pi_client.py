import numpy as np
import pyaudio as pa
import socket
import pickle
import queue as q
import asyncio
from asyncio.exceptions import CancelledError
from typing import Any, AsyncGenerator
import multiprocessing as mp
from threading import Event, Thread
import time

TIME_SLEEP = 0.02

class PiClient:

    def __init__(self,
                 host : str,
                 port = int):
        self.host = host
        self.port = port
        self.recv_queue = mp.Queue()
        self.p_send_pipe, self.c_send_pipe = mp.Pipe()
        self.shutdown_event = mp.Event()
        self.recv_thread = Thread(target=self._recv_worker)
        self.send_thread = Thread(target=self._send_worker)
        #self.recv_future = executor.submit(self._recv_worker)
        #self.send_future = executor.submit(self._send_worker)

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.connect((self.host, self.port))
        print(f"Server connecting to {self.host}:{self.port}")
        self.recv_thread.start()
        self.send_thread.start()
        # start receive loop
        #self.receive()

    def _send_worker(self):
        while not self.shutdown_event.is_set():
            if self.c_send_pipe:
                if self.c_send_pipe.poll(0.01):
                    try:
                        data = self.c_send_pipe.recv()
                        self._send(data)
                    except Exception as e:
                        print(f"Error sending data: {e}")
            time.sleep(TIME_SLEEP)

    def _recv_worker(self):
        while not self.shutdown_event.is_set():
            try:
                time.sleep(0.1)
                self.receive()
                #recv_data = self.recv_queue.get(timeout=0.1):
            except q.Empty:
                continue
            except KeyboardInterrupt:
                self.shutdown_event.set()
                break
            except Exception as e:
                print(f"Error receiving data: {e}")

    def send(self, obj):
        if self.p_send_pipe:
            self.p_send_pipe.send(obj)
    
    def _send(self, obj):
        val = pickle.dumps(obj)
        length = len(val)
        length = str(length).rjust(8,"0")
        self.socket.sendall(bytes(length, "utf-8"))
        self.socket.sendall(val)

    def receive(self):
        print("PiClient.receive: Starting receive")
        try:
            length = self.socket.recv(8)
            if not length or length == "":
                return

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
            pass
    
    def finalize(self):
        self.shutdown_event.set()
        self.p_send_pipe.close()
        self.c_send_pipe.close()
        if self.socket:
            self.socket.close()

    def next_bytes(self):
        #print("Starting next bytes")
        if not self.recv_queue.empty():
            return self.recv_queue.get()
        else:
            return None
    
