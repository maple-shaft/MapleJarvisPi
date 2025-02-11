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

TIME_SLEEP = 2.0

class PiClient:

    def __init__(self,
                 host : str,
                 port = int,
                 shutdown_event : mp.Event = mp.Event(),
                 debug : bool = False,
                 executor = None):
        self.host = host
        self.port = port
        self.recv_queue = mp.Queue()
        self.send_queue = mp.Queue()
        self.shutdown_event = shutdown_event
        if executor:
            self.executor = executor
        else:
            self.client_worker_thread = Thread(target=self._client_worker)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.debug = debug,
        self.executor = executor

    def start(self):
        if self.debug:
            print("PiClient.start: About to start the client_worker thread")
        #self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #self.socket.connect((self.host, self.port))
        #print(f"Server connecting to {self.host}:{self.port}")
        if self.executor:
            self.executor.submit(self._client_worker)
        else:
            self.client_worker_thread.start()

    def _client_worker(self):
        if self.debug:
            print("PiClient.client_worker: Starting client worker")
        self.socket.connect((self.host, self.port))
        if self.debug:
            print(f"PiClient.client_worker: Server connected to {self.host}:{self.port}")
        while not self.shutdown_event.is_set():
            try:
                time.sleep(TIME_SLEEP)
                if self.debug:
                    print("PiClient.client_worker: About to check if send_queue is empty")
                if self.send_queue.empty():
                    if self.debug:
                        print("PiClient.client_worker: Send queue is empty, continuing...")
                    continue
                if self.debug:
                    print("PiClient.client_worker: About to get latest object from send queue")
                send_data = self.send_queue.get(timeout=1.0)
                if self.debug:
                    print(f"PiClient.client_worker: Got send_data of type {type(send_data)}")
                self._send(send_data)
            except q.Empty:
                pass
            except Exception as e:
                print(f"PiClient._client_worker: Error sending data: {e}")
            
            try:
                self.receive()
            except Exception as e:
                print(f"PiClient._client_worker: Error receiving data: {e}")

            time.sleep(TIME_SLEEP)

    def send(self, obj):
        self.send_queue.put(obj)
    
    def _send(self, obj):
        if self.debug:
            print(f"PiClient._send: Starting _send of obj of type {type(obj)}")
        val = pickle.dumps(obj)
        length = len(val)
        length = str(length).rjust(8,"0")
        if self.debug:
            print(f"PiClient._send: Length of pickled obj data to send = {length}")
        self.socket.sendall(bytes(length, "utf-8"))
        if self.debug:
            print("PiClient._send: Sent the length bytes to server, now send the pickled data")
        self.socket.sendall(val)
        if self.debug:
            print("PiClient._send: Sent the pickled data to the server.")

    def receive(self):
        #print("PiClient.receive: Starting receive")
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
            raise
    
    def finalize(self):
        self.shutdown_event.set()
        if self.socket:
            self.socket.close()

    def next_bytes(self):
        #print("Starting next bytes")
        if not self.recv_queue.empty():
            return self.recv_queue.get()
        else:
            return None
    
