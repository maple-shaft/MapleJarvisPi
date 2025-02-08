import numpy as np
import openwakeword as oww
from audio_input import MicrophoneAudioInput
from audio_to_text import AudioRecorder
import wave

def get_wav_as_bytearray(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            return bytearray(wf.readframes(wf.getnframes()))
    except wave.Error as e:
        print(f"Error reading WAV file: {e}")
        return None
    except FileNotFoundError:
         print(f"File not found: {file_path}")
         return None

# Example usage:
file_path = '/home/pi/testt.wav' # Replace with the actual path to your WAV file
audio_data = get_wav_as_bytearray(file_path)

if audio_data:
    print(f"WAV file read successfully. Size: {len(audio_data)} bytes")
    # You can now work with the audio_data bytearray
else:
    print("Failed to read WAV file.")

def init_wakewords():
    try:
        return oww.Model(
            wakeword_models=["hey_jarvis_v0.1.tflite"]
        )
    except Exception as e:
        print(f"Exception encountered {e}")


def process_wakeword(data : bytearray, openwakeword : oww.Model):
    pcm = np.frombuffer(data, dtype=np.int16)
    
    prediction = openwakeword.predict(pcm)
    print(f"prediction: {prediction}")
    max_score = -1
    max_index = -1
    wakewords_in_prediction = len(openwakeword.prediction_buffer.keys())
    if wakewords_in_prediction:
        for idx, mdl in enumerate(openwakeword.prediction_buffer.keys()):
            scores = list(openwakeword.prediction_buffer[mdl])
            #if scores[-1] > 0.01:
            print(f"Score = {scores[-1]}")
                # Test code
                #testb = pcm.tobytes()
                #import wave
                #with wave.open("/tmp/maybetest.wav", "wb") as wv:
                #    wv.setnchannels(1)
                #    wv.setframerate(16000)
                #    wv.setsampwidth(2)
                #    wv.writeframes(testb)
            if scores[-1] >= 0.005 and scores[-1] > max_score:
                max_score = scores[-1]
                max_index = idx
        return max_index  
    else:
        return -1
    
def preprocess(chunk, target_sample_rate) -> bytes:
    from scipy import signal as sig
    sample_rate = 16000
    if isinstance(chunk, np.ndarray):
        if chunk.ndim == 2:
            chunk = np.mean(chunk, axis=1)

        # resample if necessary
        if sample_rate != target_sample_rate:
            num_samples = int(len(chunk) * target_sample_rate / sample_rate)
            chunk = sig.resample(chunk, num_samples)

        # Ensure it is fp16
        chunk = chunk.astype(np.int16)
    else:
        # chunk must be bytes
        chunk = np.frombuffer(chunk, dtype=np.int16)
        # resample if necessary
        if sample_rate != target_sample_rate:
            num_samples = int(len(chunk) * target_sample_rate / sample_rate)
            chunk = sig.resample(chunk, num_samples)
            # Ensure it is fp16
            chunk = chunk.astype(np.int16)
    return chunk.tobytes()

o = init_wakewords()

import queue as q
buffer = bytearray()
audio_queue = q.Queue()
processed_data = preprocess(audio_data, 16000)
buffer += processed_data
max_score = -1
while len(buffer) >= 512:
    to_process = buffer[:512]
    buffer = buffer[512:]
    audio_queue.put(to_process)
    ret = process_wakeword(to_process, o)
    print(f"process wakeword returned {ret} for this chunk")
    max_score = ret if ret > max_score else max_score

print(f"Max score: {max_score}")
print("Loaded up the wav file data in a queue")


