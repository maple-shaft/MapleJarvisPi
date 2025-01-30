import openwakeword as oww
import queue
import time
import datetime
import threading
import collections
import struct
import signal as system_signal
import torch.multiprocessing as mp
import torch
import numpy as np
import copy
from typing import List, Optional
from audio_input import AudioInput
from transcribe import TranscriptionWorker

class AudioRecorder:

    def __init__(
        self,
        audio_input : AudioInput,
        buffer_size : int = 512,
        sample_rate : int = 16000,
        device : str = "cpu",
        use_microphone = True,
        wakewords = "hey jarvis",
        silero_vad_model = None
        ):

        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.device = device
        self.pre_recording_buffer_duration : float = 1.0
        self.language = None
        self.realtime_batch_size : int = 16
        self.audio_input = audio_input
        self.silero_vad_model = silero_vad_model
        self.on_recording_stop = None

        # Wakeword detection
        self.openwakeword : oww.Model | None = None
        self.openwakeword_buffer_size = None
        self.openwakeword_frame_rate = None
        self.wakeword_backend = "openwakeword"
        self.wakeword_activation_delay : float = (0.0)
        self.wakeword_timeout : float = 5.0
        self.wakeword_buffer_duration : float = 0.1
        self.wakeword_detected = False
        self.wakeword_detect_time = 0
        self.on_wakeword_detected = None
        self.on_wakeword_timeout = None
        self.on_wakeword_detection_start = None
        self.on_wakeword_detection_end = None
        self.on_recorded_chunk = None
        self.wakewords_list = [
            word.strip() for word in wakewords.lower().split(',')
        ]
        self.wakewords_sensitivity = 0.6
        self.wakewords_sensitivities = [
            float(self.wakewords_sensitivity)
            for _ in range(len(self.wakewords_list))
        ]

        self.init_wakewords()

        # Callbacks
        self.on_recording_start = None

        self.listen_start = 0 # timing used in wakeup
        self.recording_stop_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.silero_sensitivity = 0.4
        self.silero_deactivity_detection = False
        self.input_device_index : int | None = None
        self.min_length_of_recording = 0.5
        self.beam_size_realtime = 3
        self.suppress_tokens: Optional[List[int]] = [-1]
        self.early_transcription_on_silence = 0
        self.post_speech_silence_duration = 0.6
        self.transcribe_count = 0

        # Flags and state
        self.is_silero_speech_active = False
        self.allowed_to_early_transcribe = True
        self.start_recording_on_voice_activity = True
        self.stop_recording_on_voice_deactivity = True
        self.is_shutdown = False
        self.is_recording = False
        self.is_running = True
        self.state = "inactive"
        self.realtime_model_type = "tiny"
        self.use_microphone = use_microphone
        
        # Audio data
        self.audio = None
        self.audio_queue = mp.Queue()
        self.audio_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) * self.pre_recording_buffer_duration)
        )
        self.last_words_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) * 0.3)
        )
        self.frames = []

        # Events
        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event() # Prolly should move this to a section for transcription stuff
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.shutdown_event = threading.Event()

        try:
            # Only set the start method if it hasn't been set already
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            print(f"Start method has already been set. Details: {e}")

        # Threads
        self.recording_thread = None
        self.realtime_thread = None
        self.reader_process = None
        self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()
        self.parent_stdout_pipe, child_stdout_pipe = mp.Pipe()
        self.shutdown_lock = threading.Lock()
        self.transcription_lock = threading.Lock()

        self.transcript_process = self._start_thread(
            target=AudioRecorder._transcription_worker,
            args=(
                child_transcription_pipe,               # conn
                child_stdout_pipe,                      # stdout_pipe
                "tiny",                                 # model_path
                None,                                   # download_root
                "default",                              # compute_type
                0,                                      # gpu_device_index
                self.device,                            # device
                self.main_transcription_ready_event,    # ready_event
                self.shutdown_event,                    # shutdown_event
                self.interrupt_stop_event,              # interrupt_stop_event
                5,                                      # beam_size
                None,                                   # initial_prompt
                [-1],                                   # suppress_tokens
                16                                      # batch_size
            )
        )

        # Start audio data reading process
        if self.use_microphone:
            print("Initializing audio recording"
                         " (creating pyAudio input stream,"
                         f" sample rate: {self.sample_rate}"
                         f" buffer size: {self.buffer_size}"
                         )
            self.reader_process = self._start_thread(
                target=AudioRecorder._audio_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.audio_input
                )
            )
        
        # Setup voice activity detection model Silero VAD
        try:
            if not self.silero_vad_model:
                self.silero_vad_model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    verbose=False,
                    onnx=False
                )
        except Exception as e:
            print(f"Error initializing Silero VAD "
                              f"voice activity detection engine: {e}"
                              )
            raise

        print("Silero VAD voice activity detection "
                      "engine initialized successfully"
                      )
        
        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def _start_thread(self, target=None, args=()) -> threading.Thread:
        """If linux uses standard threads, otherwise uses pytorch multiprocessing"""
        thread = threading.Thread(target = target, args = args)
        thread.daemon = True
        thread.start()
        return thread

    def _transcription_worker(*args, **kwargs):
        worker = TranscriptionWorker(*args, **kwargs)
        worker.run()

    def init_wakewords(self):
        try:
            self.openwakeword = oww.Model(
                wakeword_models=["hey_jarvis_v0.1.tflite"]
            )
        except Exception as e:
            print(f"Exception encountered {e}")

    @staticmethod
    def _audio_worker(
        audio_queue : queue.Queue,
        target_sample_rate : int,
        buffer_size : int,
        device_index : int | None,
        shutdown_event : threading.Event,
        interrupt_stop_event : threading.Event,
        audio_input : AudioInput):

        if __name__ == '__main__':
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
        
        audio_input.setup()

        buffer = bytearray()
        silero_buffer_size = 2 * buffer_size
        time_since_last_buffer_message = 0
        try:
            while not shutdown_event.is_set():
                try:
                    data = audio_input.read_chunk()
                    if data:
                        processed_data = audio_input.preprocess(data, target_sample_rate=target_sample_rate)

                        buffer += processed_data
                        while len(buffer) >= silero_buffer_size:
                            to_process = buffer[:silero_buffer_size]
                            buffer = buffer[silero_buffer_size:]

                            # DAB: Debug procedure to see what the hell is captured in this audio
                            #from pydub import AudioSegment
                            #from scipy.io import wavfile
                            #import uuid
                            #try:
                                # Normalize and convert NumPy array to int16 PCM
                            #    audio_wav = np.frombuffer(to_process, dtype=np.int16)
                            #    # Create an in-memory buffer for raw audio
                            #    wavfile.write(filename=f"/tmp/wav_{uuid.uuid4()}", rate = 16000, data = audio_wav)
                            #    # Convert raw audio into an AudioSegment   
                            #except Exception as e:
                            #    print(f"Exception writing audio file for testing: {e}")

                            # feed to the audio_queue
                            if time_since_last_buffer_message:
                                time_passed = time.time() - time_since_last_buffer_message
                                if time_passed > 1:
                                    #print("still writing audio into the audio_queue")
                                    time_since_last_buffer_message = time.time()
                            else:
                                time_since_last_buffer_message = time.time()

                            audio_queue.put(to_process)
                except OSError as oe:
                    print(f"OSError encountered: {oe}")
                    continue
                except Exception as e:
                    print(f"Exception encountered: {e}")
                    continue
        except KeyboardInterrupt:
            interrupt_stop_event.set()
            print("Audio data worker process finished due to KeyboardInterrupt")
        finally:
            # After recording stops, feed remaining buffer data
            if buffer:
                audio_queue.put(bytes(buffer))
            try:
                if audio_input:
                    audio_input.cleanup()
            except Exception as e:
                print(f"Exception encountered during audio_input cleanup {e}")

    def wakeup(self):
        """Wakeup the audio_input"""
        self.listen_start = time.time()

    def start(self):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        # Ensure there's a minimum interval
        # between stopping and starting recording
        if (time.time() - self.recording_stop_time
                < 0.0):
            print("Attempted to start recording "
                         "too soon after stopping."
                         )
            return self

        print("recording started")
        self._set_state("recording")
        self.wakeword_detected = False
        self.wakeword_detect_time = 0
        self.frames = []
        self.is_recording = True
        self.recording_start_time = time.time()
        self.is_silero_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            self.on_recording_start()

        return self

    def stop(self):
        """
        Stops recording audio.
        """
        # Ensure there's a minimum interval
        # between starting and stopping recording
        if (time.time() - self.recording_start_time
                < self.min_length_of_recording):
            print("Attempted to stop recording too soon after starting")
            return self

        self.is_recording = False
        self.recording_stop_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        if self.on_recording_stop:
            self.on_recording_stop()

        return self
    
    def abort(self):
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self._set_state("inactive")
        self.interrupt_stop_event.set()
        self.was_interrupted.wait()
        self.was_interrupted.clear()

    def listen(self):
        """
        Puts recorder in immediate "listen" state.
        This is the state after a wake word detection, for example.
        The recorder now "listens" for voice activation.
        Once voice is detected we enter "recording" state.
        """
        self.listen_start = time.time()
        self._set_state("listening")
        self.start_recording_on_voice_activity = True

    def wait_audio(self):
        """Waits for the start and completion of the recording process"""
        try:
            if self.listen_start == 0:
                self.listen_start = time.time()

            # If not already recording, wait for voice activity to start
            if not self.is_recording and not self.frames:
                self._set_state("listening")
                self.start_recording_on_voice_activity = True

                print("Waiting for recording to start...")
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout = 0.04):
                        break
            # if recording is ongoing, then wait for voice inactivity
            if self.is_recording:
                self.stop_recording_on_voice_deactivity = True
                print("Waiting for recording to stop...")
                while not self.interrupt_stop_event.is_set():
                    if (self.stop_recording_event.wait(timeout = 0.02)):
                        break

            # Convert recorded frames to the appropriate audio format
            audio_array = np.frombuffer(b"".join(self.frames), dtype=np.int16)
            self.audio = audio_array.astype(np.float32) / 32768.0
            self.frames.clear()

            # Reset recording related timestamps
            self.recording_stop_time = 0
            self.listen_start = 0
            self._set_state("inactive")
        except Exception as e:
            print(f"Exception encounterd in wait_audio {e}")
            self.shutdown()
            raise

    def shutdown(self):
        """Safely shut down the audio recording"""
        with self.shutdown_lock:
            if self.is_shutdown:
                return
            
            print("Shutting down AudioRecorder")
            self.is_shutdown = True
            self.start_recording_event.set()
            self.stop_recording_event.set()
            self.is_recording = False
            self.is_running = False
            if self.recording_thread:
                self.recording_thread.join()

            if self.reader_process:
                self.reader_process.join(timeout=10)
                if self.reader_process.is_alive():
                    print("Reader process didnt terminate, forcefully terminate it")
                    self.reader_process.terminate()
            
            import gc
            gc.collect()

    def _preprocess_output(self, text, preview=False):
        """
        Preprocesses the output text by removing any leading or trailing
        whitespace, converting all whitespace sequences to a single space
        character, and capitalizing the first character of the text.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        import re
        text = re.sub(r'\s+', ' ', text.strip())

        if text:
            text = text[0].upper() + text[1:]

        # Ensure the text ends with a proper punctuation
        # if it ends with an alphanumeric character
        if not preview:
            if text and text[-1].isalnum():
                text += '.'
        return text

    def _process_wakeword(self, data : bytearray):
        """
        Processes audio data to detect wake words.
        """
        
        pcm = np.frombuffer(data, dtype=np.int16)
        
        prediction = self.openwakeword.predict(pcm)
        #print(f"prediction: {prediction}")
        max_score = -1
        max_index = -1
        wakewords_in_prediction = len(self.openwakeword.prediction_buffer.keys())
        self.wakewords_sensitivities
        if wakewords_in_prediction:
            for idx, mdl in enumerate(self.openwakeword.prediction_buffer.keys()):
                scores = list(self.openwakeword.prediction_buffer[mdl])
                if scores[-1] > 0.01:
                    print(f"Score = {scores[-1]}")
                    # Test code
                    testb = pcm.tobytes()
                    import wave
                    with wave.open("/tmp/maybetest.wav", "wb") as wv:
                        wv.setnchannels(1)
                        wv.setframerate(16000)
                        wv.setsampwidth(2)
                        wv.writeframes(testb)
                if scores[-1] >= self.wakewords_sensitivity and scores[-1] > max_score:
                    max_score = scores[-1]
                    max_index = idx
            return max_index  
        else:
            return -1
    
    def transcribe(self):
        """
        Transcribes audio captured by this class instance using the
        `faster_whisper` model.

        Automatically starts recording upon voice activity if not manually
          started using `recorder.start()`.
        Automatically stops recording upon voice deactivity if not manually
          stopped with `recorder.stop()`.
        Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): Callback function
              to be executed when transcription is ready.
            If provided, transcription will be performed asynchronously,
              and the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously,
              and the result will be returned.

        Returns (if no callback is set):
            str: The transcription of the recorded audio.

        Raises:
            Exception: If there is an error during the transcription process.
        """

        self._set_state("transcribing")
        audio_copy = copy.deepcopy(self.audio)
        with self.transcription_lock:
            try:
                if self.transcribe_count == 0:
                    self.parent_transcription_pipe.send((audio_copy, self.language))
                    self.transcribe_count += 1

                while self.transcribe_count > 0:
                    status, result = self.parent_transcription_pipe.recv()
                    self.transcribe_count -= 1

                self.allowed_to_early_transcribe = True
                self._set_state("inactive")
                if status == 'success':
                    ret_audio, ret_lang = result
                    self.last_transcription_bytes = copy.deepcopy(audio_copy)                    
                    return ret_audio
                else:
                    raise Exception(result)
            except Exception as e:
                print(f"Error during transcription: {str(e)}")
                raise e

    def get_audio_for_transcription(self):
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        try:
            self.wait_audio()
        except KeyboardInterrupt:
            print("KeyboardInterrupt in text() method")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

        if self.is_shutdown or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return None

        return self.transcribe()

    def _recording_worker(self):
        """The main recording worker method, records audio consistently"""
        import numpy as np
        last_inner_try_time = 0
        try:
            time_since_last_buffer_message = 0
            was_recording = False
            delay_was_passed = False
            wakeword_detected_time = None
            wakeword_samples_to_remove = None
            self.allowed_to_early_transcribe = True

            print("Initialized recording worker")
            while self.is_running:
                if last_inner_try_time:
                    last_processing_time = time.time() - last_inner_try_time
                    if last_processing_time > 0.1:
                        print("Processing took a bit too long...")
                last_inner_try_time = time.time()
                try:
                    try:
                        data = self.audio_queue.get(timeout=0.01)
                        self.last_words_buffer.append(data)
                    except queue.Empty:
                        if not self.is_running:
                            print("Recording worker: Not running, breaking loop")
                            break
                        continue
                    #print("DAB: Found data in the audio queue in _recording_worker")
                    # Is there a callback defined for on_recorded_chunk?
                    if self.on_recorded_chunk:
                        self.on_recorded_chunk(data)

                    # Check handle buffer overflow, assume True for now
                    if self.audio_queue.qsize() > 100:  # allowed latency limit
                        print("Recording Worker: queue size exceeds limit")

                    while self.audio_queue.qsize() > 100:
                        data = self.audio_queue.get()
                except BrokenPipeError:
                    print("Recording Worker: broken pipe error")
                    self.is_running = False
                    break

                if time_since_last_buffer_message:
                    time_passed = time.time() - time_since_last_buffer_message
                    if time_passed > 1:
                        time_since_last_buffer_message = time.time()
                else:
                    time_since_last_buffer_message = time.time()

                failed_stop_attempt = False

                #print("DAB: _recording_worker - Check if self.is_recording.  If done recording then we can look for wakewords")
                if not self.is_recording:
                    #print("not recording...")
                    time_since_listen_start = (time.time() - self.listen_start if self.listen_start else 0)
                    wakeword_activation_delay_passed = (
                        time_since_listen_start > self.wakeword_activation_delay
                    )
                    #print(f"wakeword_activation_delay_passed {wakeword_activation_delay_passed} and delay_was_passed {delay_was_passed}")
                    if wakeword_activation_delay_passed and not delay_was_passed:
                        #print(f"self.wakeword_activation_delay {self.wakeword_activation_delay}")
                        if self.wakeword_activation_delay:
                            if self.on_wakeword_timeout:
                                self.on_wakeword_timeout()
                    delay_was_passed = wakeword_activation_delay_passed
                    
                    # Set state
                    if not self.recording_stop_time:
                        if wakeword_activation_delay_passed and not self.wakeword_detected:
                            self._set_state("wakeword")
                        else:
                            if self.listen_start:
                                self._set_state("listening")
                            else:
                                self._set_state("inactive")
                    if wakeword_activation_delay_passed:
                        try:
                            #TODO: This is supposed to return an index where 0 means no wakeword was found, and more
                            # than 0 means that it was found. I think it expects this number to represent a number of 
                            # word samples that should be removed from the audio buffer?
                            wakeword_index = self._process_wakeword(data)
                        except struct.error:
                            print("Recording Worker: Error unpacking audio data")
                            continue
                        except Exception as e:
                            print(f"Recording Worker: Wakeword processing error {e}")
                            continue

                        # if a wakeword is detected
                        if wakeword_index >= 0:
                            wakeword_samples_to_remove = int(self.sample_rate * self.wakeword_buffer_duration)
                            self.wakeword_detected = True
                            if self.on_wakeword_detected:
                                self.on_wakeword_detected()

                    # Check for voice activity to trigger start of recording
                    if (not wakeword_activation_delay_passed and self.start_recording_on_voice_activity) or self.wakeword_detected:
                        if self._is_voice_active():
                            self.start()
                            self.start_recording_on_voice_activity = False
                            # Add the buffered audio
                            # to the recording frames
                            self.frames.extend(list(self.audio_buffer))
                            self.audio_buffer.clear()
                            self.silero_vad_model.reset_states()
                        else:
                            data_copy = data[:]
                            self._check_voice_activity(data_copy)
                    self.speech_end_silence_start = 0
                else:
                    # If we are currently recording
                    if wakeword_samples_to_remove and wakeword_samples_to_remove > 0:
                        # Remove samples from the beginning of self.frames
                        samples_removed = 0
                        while wakeword_samples_to_remove > 0 and self.frames:
                            frame = self.frames[0]
                            frame_samples = len(frame) // 2  # Assuming 16-bit audio
                            if wakeword_samples_to_remove >= frame_samples:
                                self.frames.pop(0)
                                samples_removed += frame_samples
                                wakeword_samples_to_remove -= frame_samples
                            else:
                                self.frames[0] = frame[wakeword_samples_to_remove * 2:]
                                samples_removed += wakeword_samples_to_remove
                                samples_to_remove = 0
                        
                        wakeword_samples_to_remove = 0

                    # Stop the recording if silence is detected after speech
                    if self.stop_recording_on_voice_deactivity:
                        is_speech = self.silero_deactivity_detection and self._is_silero_speech(data)
                        
                        if not self.speech_end_silence_start:
                            str_speech_end_silence_start = "0"
                        else:
                            str_speech_end_silence_start = datetime.datetime.fromtimestamp(self.speech_end_silence_start).strftime('%H:%M:%S.%f')[:-3]
                        
                        if not is_speech:
                            # Voice deactivity was detected, so we start
                            # measuring silence time before stopping recording
                            if self.speech_end_silence_start == 0 and \
                                (time.time() - self.recording_start_time > self.min_length_of_recording):
                                self.speech_end_silence_start = time.time()
                            if self.speech_end_silence_start and self.early_transcription_on_silence and len(self.frames) > 0 and \
                                (time.time() - self.speech_end_silence_start > self.early_transcription_on_silence) and \
                                self.allowed_to_early_transcribe:
                                    self.transcribe_count += 1
                                    audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                                    audio = audio_array.astype(np.float32) / 32768.0
                                    self.parent_transcription_pipe.send((audio, self.language))
                                    self.allowed_to_early_transcribe = False
                        else:
                            if self.speech_end_silence_start:
                                self.speech_end_silence_start = 0
                                self.allowed_to_early_transcribe = True
                        # Wait for silence to stop recording after speech
                        if self.speech_end_silence_start and time.time() - \
                                self.speech_end_silence_start >= \
                                self.post_speech_silence_duration:
                            # Get time in desired format (HH:MM:SS.nnn)
                            silence_start_time = datetime.datetime.fromtimestamp(self.speech_end_silence_start).strftime('%H:%M:%S.%f')[:-3]

                            # Calculate time difference
                            time_diff = time.time() - self.speech_end_silence_start

                            self.frames.append(data)
                            self.stop()
                            if not self.is_recording:
                                self.speech_end_silence_start = 0
                            else:
                                failed_stop_attempt = True

                if not self.is_recording and was_recording:
                    # Reset after stopping recording to ensure clean state
                    self.stop_recording_on_voice_deactivity = False

                if time.time() - self.silero_check_time > 0.1:
                    self.silero_check_time = 0

                # Handle wake word timeout (waited to long initiating
                # speech after wake word detection)
                if self.wakeword_detect_time and time.time() - \
                        self.wakeword_detect_time > self.wakeword_timeout:

                    self.wakeword_detect_time = 0
                    if self.wakeword_detected and self.on_wakeword_timeout:
                        self.on_wakeword_timeout()
                    self.wakeword_detected = False
                was_recording = self.is_recording

                if self.is_recording and not failed_stop_attempt:
                    self.frames.append(data)

                if not self.is_recording or self.speech_end_silence_start:
                    self.audio_buffer.append(data)
        except Exception as e:
            print(f"Recorder Worker: Exception encountered {e}")
            if not self.interrupt_stop_event.is_set():
                raise

    def _is_voice_active(self):
        """
        Determine if voice is active.

        Returns:
            bool: True if voice is active, False otherwise.
        """
        return self.is_silero_speech_active

    def _is_silero_speech(self, chunk):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        from scipy import signal
        #print("DAB: Check for speech in is_silero_speech...")

        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        self.silero_working = True
        audio_chunk = np.frombuffer(chunk, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / 32768.0
        vad_prob = self.silero_vad_model(
            torch.from_numpy(audio_chunk), 16000).item()
        import math
        rd_prob = math.floor(vad_prob * 100)
        print(f"{rd_prob}")
        is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
        self.is_silero_speech_active = is_silero_speech_active
        self.silero_working = False
        #print(f"DAB: is_silero_speech_active = {is_silero_speech_active}")
        return is_silero_speech_active

    def _check_voice_activity(self, data):
        """
        Initiate check if voice is active based on the provided data.

        Args:
            data: The audio data to be checked for voice activity.
        """
        #print("DAB: Checking voice activity...")
        if not self.silero_working:
            self.silero_working = True

            # Run the intensive check in a separate thread
            threading.Thread(
                target=self._is_silero_speech,
                args=(data,)).start()

    def clear_audio_queue(self):
        """
        Safely empties the audio queue to ensure no remaining audio 
        fragments get processed e.g. after waking up the recorder.
        """
        self.audio_buffer.clear()
        try:
            while True:
                self.audio_queue.get_nowait()
        except:
            # PyTorch's mp.Queue doesn't have a specific Empty exception
            # so we catch any exception that might occur when the queue is empty
            pass

    def _is_voice_active(self):
        return self.is_silero_speech_active
    
    def _set_state(self, new_state):
        """Update the current state of the recorder and execute state-change callbacks"""
        if new_state == self.state:
            return
        old_state = self.state
        self.state = new_state

        # From state callbacks
        if old_state == "listening":
            print("Old state is 'listening'")
            # TODO: Add a callback handler execution similar to self.on_vad_detect_stop
        elif old_state == "wakeword":
            print("Old state is 'wakeword'")
            # TODO: Add a callback handler execution similar to self.on_wakeword_detection_end

        # To state callbacks
        if new_state == "listening":
            print("New state is 'listening'")
        elif new_state == "recording":
            print("New state is 'recording'")
        elif new_state == "inactive":
            print("New state is 'inactive'")
        elif new_state == "wakeword":
            print("New state is 'wakeword'")

    def __enter__(self):
        """
        Method to setup the context manager protocol.

        This enables the instance to be used in a `with` statement, ensuring
        proper resource management. When the `with` block is entered, this
        method is automatically called.

        Returns:
            self: The current instance of the class.
        """
        return self

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):
        """
        Find the position where the last 'n' characters of text1
        match with a substring in text2.

        This method takes two texts, extracts the last 'n' characters from
        text1 (where 'n' is determined by the variable 'length_of_match'), and
        searches for an occurrence of this substring in text2, starting from
        the end of text2 and moving towards the beginning.

        Parameters:
        - text1 (str): The text containing the substring that we want to find
          in text2.
        - text2 (str): The text in which we want to find the matching
          substring.
        - length_of_match(int): The length of the matching string that we are
          looking for

        Returns:
        int: The position (0-based index) in text2 where the matching
          substring starts. If no match is found or either of the texts is
          too short, returns -1.
        """

        # Check if either of the texts is too short
        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1

        # The end portion of the first text that we want to compare
        target_substring = text1[-length_of_match:]

        # Loop through text2 from right to left
        for i in range(len(text2) - length_of_match + 1):
            # Extract the substring from text2
            # to compare with the target_substring
            current_substring = text2[len(text2) - i - length_of_match:
                                      len(text2) - i]

            # Compare the current_substring with the target_substring
            if current_substring == target_substring:
                # Position in text2 where the match starts
                return len(text2) - i

        return -1
