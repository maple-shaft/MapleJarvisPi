import pyaudio
import numpy as np
from io import BytesIO
from pydub import AudioSegment

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
DESIRED_RATE = 16
SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2

class AudioInput:

    def __init__(self,
                 chunk = CHUNK,
                 format = FORMAT,
                 channels = CHANNELS,
                 desired_rate = DESIRED_RATE,
                 sample_rate = SAMPLE_RATE,
                 debug = False):
        self.sample_rate = sample_rate
        self.stream = None
        self.debug = debug

        # Constants
        self.chunk = chunk
        self.format = format
        self.channels = channels
        self.desired_rate = desired_rate

    def setup(self):
        """Initialize audio interface and setup stream"""
        raise NotImplementedError("Subclass must implement this abstract method")
        
    def read_chunk(self):
        debug = self.debug
        while not any(v := self.stream.read(1, False)):
            pass
        print(f"AudioInput.read_chunk: Read non zero chunk {v}") if debug else None
        return self.stream.read(self.chunk, exception_on_overflow=False)
    
    def preprocess(self, chunk, target_sample_rate) -> bytes:
        from scipy import signal as sig
        if isinstance(chunk, np.ndarray):
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            # resample if necessary
            if self.sample_rate != target_sample_rate:
                num_samples = int(len(chunk) * target_sample_rate / self.sample_rate)
                chunk = sig.resample(chunk, num_samples)

            # Ensure it is fp16
            chunk = chunk.astype(np.int16)
        else:
            # chunk must be bytes
            chunk = np.frombuffer(chunk, dtype=np.int16)
            # resample if necessary
            if self.sample_rate != target_sample_rate:
                num_samples = int(len(chunk) * target_sample_rate / self.sample_rate)
                chunk = sig.resample(chunk, num_samples)
                # Ensure it is fp16
                chunk = chunk.astype(np.int16)
        return chunk.tobytes()
    
    def cleanup(self):
        """Cleanup audio resources"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            if self.audio_interface:
                self.audio_interface.terminate()
                self.audio_interface = None
        except Exception as e:
            print(f"Exception encountered {e}")

class MicrophoneAudioInput(AudioInput):

    def __init__(self,
                 chunk = CHUNK,
                 format = FORMAT,
                 channels = CHANNELS,
                 desired_rate = DESIRED_RATE,
                 sample_rate = SAMPLE_RATE,
                 device_index = 2,
                 debug = False):
        super(MicrophoneAudioInput, self).__init__(chunk,format,channels,desired_rate,sample_rate,debug)
        self.device_index = device_index
        self.audio_interface = None

    def setup(self):
        """Initialize audio interface and setup stream"""
        try:
            self.audio_interface = pyaudio.PyAudio()
            self.stream = self.audio_interface.open(
                format = self.format,
                channels = self.channels,
                rate = self.sample_rate,
                input = True,
                frames_per_buffer = self.chunk,
                input_device_index = self.device_index
            )
            return True
        except Exception as e:
            print(f"Exception encountered: {e}")
            if self.audio_interface:
                self.audio_interface.terminate()
            return False

    def list_devices(self):
        try:
            self.audio_interface = pyaudio.PyAudio()
            device_count = self.audio_interface.get_device_count()
            for i in range(device_count):
                device_info = self.audio_interface.get_device_info_by_index(i)
                device_name = device_info.get("name")
                print(f"list_devices -> Device ID: {i}, Device Name: {device_name}")
                if (device_info.get("maxInputChannels", 0) > 0):
                    # Only consider devices with record capability
                    print(f"list_devices -> Recording Device Found!")

        except Exception as e:
            print(f"Exception encountered: {e}")
        finally:
            if self.audio_interface:
                self.audio_interface.terminate()

