import queue
import time
import threading
import numpy as np
import signal as system_signal

TIME_SLEEP = 0.02

class TranscriptionWorker:
    def __init__(self, conn, stdout_pipe, model_path, download_root, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens, batch_size):
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.model_path = model_path
        self.download_root = download_root
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.batch_size = batch_size
        self.queue = queue.Queue()

    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass
    
    def poll_connection(self):
        while not self.shutdown_event.is_set():
            if self.conn.poll(0.01):
                try:
                    data = self.conn.recv()
                    self.queue.put(data)
                except Exception as e:
                    print(f"Error receiving data from connection: {e}")
            else:
                time.sleep(TIME_SLEEP)

    def run(self):
        if __name__ == "__main__":
             system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
             __builtins__['print'] = self.custom_print

        self.ready_event.set()

        # Start the polling thread
        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language = self.queue.get(timeout=0.1)
                    try:
                        self.conn.send(('success', (audio, language)))
                    except Exception as e:
                        print(f"General error in transcription: {e}")
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    print("Transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    print(f"General error in processing queue item: {e}")
        finally:
            __builtins__['print'] = print  # Restore the original print function
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()  # Ensure the polling thread will stop
            polling_thread.join()  # Wait for the polling thread to finish