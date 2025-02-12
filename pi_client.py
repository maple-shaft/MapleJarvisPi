import numpy as np
import pyaudio as pa
import socket
import select
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
                 shutdown_event : Event = Event(),
                 debug : bool = False,
                 executor = None):
        self.host = host
        self.port = port
        self.recv_queue = mp.Queue()
        self.send_queue = mp.Queue()
        self.shutdown_event : Event = shutdown_event
        if executor:
            self.executor = executor
        else:
            self.client_worker_thread = Thread(target=self._client_worker)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setblocking(False)
        self.debug = debug,
        self.executor = executor

    def start(self):
        if self.debug:
            print("PiClient.start: About to start the client_worker thread")
        
        if self.executor:
            self.executor.submit(self._client_worker)
        else:
            self.client_worker_thread.start()

    def _client_worker(self):
        print("PiClient.client_worker: Starting...") if self.debug else None
        try:
            asyncio.run(self._client_worker_async())
        except Exception as e:
            print(f"PiClient.client_worker: Exception encountered {e}")

    async def _client_worker_async(self):
        if self.debug:
            print("PiClient.client_worker_async: Starting client worker")
        try:
            # self.socket.connect_ex((self.host, self.port))
            reader, writer = await asyncio.open_connection(host=self.host, port=self.port)

            print("PiClient.client_worker_async: Created Asyncio Connection") if self.debug else None
            send_coro : asyncio.Task = None
            receive_coro : asyncio.Task = None
            while not self.shutdown_event.is_set():
                await asyncio.sleep(TIME_SLEEP)
                try:
                    if send_coro is None or send_coro.done():
                        if self.debug:
                            print("PiClient.client_worker: About to check if send_queue is empty")
                        if self.send_queue.empty():
                            if self.debug:
                                print("PiClient.client_worker: Send queue is empty, continuing...") if self.debug else None
                        else:
                            if self.debug:
                                print("PiClient.client_worker: About to get latest object from send queue")
                            send_data = self.send_queue.get(timeout=0.1)
                            if self.debug:
                                print(f"PiClient.client_worker: Got send_data of type {type(send_data)}")
                            #asyncio.create_task(self._send(send_data, writer))
                            print("ABOUT TO CHECK SEND CORO!!!!!!")
                            send_coro = asyncio.create_task(self._send(send_data, writer))
                            print(f"SEND CORO IS TYPE {type(send_coro)}")
                except Exception as e:
                    print(f"PiClient.client_worker: Error sending data: {e}")
            
                try:
                    if receive_coro is None or receive_coro.done():
                        print("PiClient.client_worker: About to call self.receive()") if self.debug else None
                        #asyncio.create_task(self.receive(reader))
                        print("ABOUT TO CHECK RECEIVE_CORO!!!!!!")
                        receive_coro = asyncio.create_task(self.receive(reader))
                        print(f"RECEIVE_CORO IS TYPE {type(receive_coro)}")
                except Exception as e:
                    print(f"PiClient.client_worker: Error receiving data: {e}")

        finally:
            print("PiClient.client_worker_async: About to close the reader and writer")
            writer.close()
            await writer.wait_closed()
            #time.sleep(TIME_SLEEP)

    def send(self, obj):
        self.send_queue.put(obj)
    
    async def _send(self, obj, writer):
        try:
            if self.debug:
                print(f"PiClient._send: Starting _send of obj of type {type(obj)}")
            val = pickle.dumps(obj)
            length = len(val)
            length = str(length).rjust(8,"0")
            if self.debug:
                print(f"PiClient._send: Length of pickled obj data to send = {length}")

            writer.write(bytes(length, "utf-8"))
            await writer.drain()
            if self.debug:
                print("PiClient._send: Sent the length bytes to server, now send the pickled data")
            writer.write(val)
            await writer.drain()
            if self.debug:
                print("PiClient._send: Sent the pickled data to the server.")
        except Exception as e:
            print(f"PiClient._send: Exception encountered {e}")
            raise

    async def receive(self, reader):
        print("PiClient.receive: Starting receive") if self.debug else None
        try:
            length = None
            length = await reader.read(8)
            if not length or length == "":
                return

            length = int(length.decode("utf-8"))
            full_length = length
            message = None
            while length > 0:
                chunk_len = min(128, length)
                length -= chunk_len
                chunk = await reader.read(chunk_len)
                if message is None:
                    message = chunk
                else:
                    message = message + chunk
            while len(message) < full_length:
                chunk_len = min(128, full_length - len(message))
                chunk = await reader.read(chunk_len)
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
    
