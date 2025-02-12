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
import webrtcvad
from typing import List, Optional
from audio_input import AudioInput
from concurrent.futures import ProcessPoolExecutor

class AudioRecorder:

    def __init__(
        self,
        audio_input : AudioInput,
        buffer_size : int = 512,
        sample_rate : int = 16000,
        device : str = "cpu",
        use_microphone = True,
        wakewords = "hey jarvis",
        silero_vad_model = None,
        wakeword_timeout : float = 5.0,
        wakeword_buffer_duration : float = 0.1,
        wakewords_sensitivity = 0.5,
        wakeword_activation_delay = 0.0,
        min_length_of_recording = 0.5, # DAB: Original value 0.5
        start_recording_on_voice_activity = True,
        stop_recording_on_voice_deactivity = True,
        prerecording_buffer_duration : float = 1.0,
        post_speech_silence_duration : float = 0.6,
        webrtc_sensitivity = 3,
        on_recording_start = None,
        on_recording_stop = None,
        on_recorded_chunk = None,
        on_wakeword_detected = None,
        on_wakeword_timeout = None,
        on_wakeword_detection_start = None,
        on_wakeword_detection_end = None,
        ctx = None,
        executor = None,
        debug = False,
        shutdown_event = None
        ):

        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.device = device
        self.pre_recording_buffer_duration : float = prerecording_buffer_duration
        self.language = None
        self.realtime_batch_size : int = 16
        self.audio_input = audio_input
        self.silero_vad_model = silero_vad_model
        self.on_recording_stop = on_recording_stop
        self.on_recording_start = on_recording_start
        self.on_recorded_chunk = on_recorded_chunk

        self.min_length_of_recording = min_length_of_recording

        self.ctx = ctx
        self.executor = executor

        # Wakeword detection
        self.openwakeword : oww.Model | None = None
        self.openwakeword_buffer_size = None
        self.openwakeword_frame_rate = None
        self.wakeword_backend = "openwakeword"
        self.wakeword_activation_delay : float = wakeword_activation_delay
        self.wakeword_timeout : float = wakeword_timeout
        self.wakeword_buffer_duration : float = wakeword_buffer_duration
        self.wakeword_detected = False
        self.wakeword_detect_time = 0
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        
        self.wakewords_list = [
            word.strip() for word in wakewords.lower().split(',')
        ]
        self.wakewords_sensitivity = wakewords_sensitivity
        self.wakewords_sensitivities = [
            float(self.wakewords_sensitivity)
            for _ in range(len(self.wakewords_list))
        ]

        self.init_wakewords()

        # Callbacks
        self.on_recording_start = on_recording_start

        self.listen_start = 0 # timing used in wakeup
        self.recording_stop_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.silero_sensitivity = 0.5
        self.silero_deactivity_detection = False
        self.input_device_index : int | None = None
        self.suppress_tokens: Optional[List[int]] = [-1]
        self.early_transcription_on_silence = 0
        self.post_speech_silence_duration = post_speech_silence_duration
        self.is_webrtc_speech_active = False

        # Flags and state
        self.debug = debug
        self.is_silero_speech_active = False
        self.start_recording_on_voice_activity = start_recording_on_voice_activity
        self.stop_recording_on_voice_deactivity = stop_recording_on_voice_deactivity
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

        self.frames = []

        # Events
        
        self.interrupt_stop_event = mp.Event() if not ctx else ctx.Event()
        self.was_interrupted = mp.Event() if not ctx else ctx.Event()
        self.start_recording_event = mp.Event() if not ctx else ctx.Event()
        self.stop_recording_event = mp.Event() if not ctx else ctx.Event()
        if not shutdown_event:
            self.shutdown_event = mp.Event() if not ctx else ctx.Event()
        else:
            self.shutdown_event = shutdown_event

        # Threads
        self.recording_thread = None
        self.realtime_thread = None
        self.reader_process = None
        self.shutdown_lock = threading.Lock()

        print("KAGNLNSAGLANGLANGOIEGNO$JEN$(#J($JF)$J)FJ$#)FJ#JFNVCKMN$#)GF#")
        # Start audio data reading process
        if self.use_microphone:
            print("AudioRecorder.__init__: Initializing audio recording"
                         " (creating pyAudio input stream,"
                         f" sample rate: {self.sample_rate}"
                         f" buffer size: {self.buffer_size}"
                         )
            if self.executor:
                print("We have an executor, submit audio_worker to it")
                self.reader_process = self.executor.submit(
                     AudioRecorder._audio_worker,
                     self.audio_queue,
                     self.sample_rate,
                     self.buffer_size,
                     self.input_device_index,
                     self.shutdown_event,
                     self.interrupt_stop_event,
                     self.audio_input
                )
            else:
                print("NO EXECUTOR")
                raise Exception()
        
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
            print(f"AudioRecorder.__init__: Error initializing Silero VAD "
                              f"voice activity detection engine: {e}"
            )
            raise

        print("AudioRecorder.__init__: Silero VAD voice activity detection "
                      "engine initialized successfully"
        )

        # Setup voice activity detection model WebRTC
        try:
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)
        except Exception as e:
            raise
        
        # Start the recording worker thread
        if self.executor:
            print("AudioRecorder.__init__: About to start recording_thread") if self.debug else None
            self.recording_thread = self.executor.submit(self._recording_worker)
        else:
            self.recording_thread = threading.Thread(self._recording_worker)
            self.recording_thread.daemon = False
            self.recording_thread.start()

    def _start_thread(self, target=None, args=(), executor = None, daemon = False):
        """If linux uses standard threads, otherwise uses pytorch multiprocessing"""
        #thread = threading.Thread(target = target, args = args)
        #thread.daemon = True

        if executor:
            thread = self.executor.submit(fn=target)
        else:
            thread = threading.Thread(target=target, args = args)
            thread.daemon = daemon
            thread.start()
        return thread

    def init_wakewords(self):
        try:
            self.openwakeword = oww.Model(
                wakeword_models=["hey_jarvis_v0.1.tflite"]
            )
        except Exception as e:
            print(f"AudioRecorder.init_wakewords: Exception encountered in init_wakewords: {e}")

    def debug_audio(self, data):
        # DAB: Debug procedure to see what the hell is captured in this audio
        from pydub import AudioSegment
        from scipy.io import wavfile
        import uuid
        try:
            # Normalize and convert NumPy array to int16 PCM
            audio_wav = np.frombuffer(data, dtype=np.int16)
            # Create an in-memory buffer for raw audio
            wavfile.write(filename=f"/tmp/wav_{uuid.uuid4()}", rate = 16000, data = audio_wav)
            # Convert raw audio into an AudioSegment   
        except Exception as e:
            print(f"Exception writing audio file for testing: {e}")

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

        #print("AudioRecorder.audio_worker: About to setup audio_input")
        audio_input.setup()

        buffer = bytearray()
        silero_buffer_size = 2 * buffer_size
        time_since_last_buffer_message = 0
        try:
            while not shutdown_event.is_set():
                try:
                    #print("AudioRecorder.audio_worker: continuing while loop, try to read a chunk")
                    data = audio_input.read_chunk()
                    if data:
                        #print("AudioRecorder.audio_worker: data is found!")
                        processed_data = audio_input.preprocess(data, target_sample_rate=target_sample_rate)

                        buffer += processed_data
                        while len(buffer) >= silero_buffer_size:
                            to_process = buffer[:silero_buffer_size]
                            buffer = buffer[silero_buffer_size:]

                            # DAB: Debug procedure to see what the hell is captured in this audio
                            #debug_audio(to_process)
                            
                            # feed to the audio_queue
                            if time_since_last_buffer_message:
                                time_passed = time.time() - time_since_last_buffer_message
                                if time_passed > 2: # DAB: Original value is 1
                                    #print("still writing audio into the audio_queue")
                                    time_since_last_buffer_message = time.time()
                            else:
                                time_since_last_buffer_message = time.time()

                            audio_queue.put(to_process)
                except OSError as oe:
                    print(f"OSError encountered in audio_worker: {oe}")
                    continue
                except Exception as e:
                    print(f"Exception encountered in audio_worker: {e}")
                    continue
        except KeyboardInterrupt:
            interrupt_stop_event.set()
            print("Audio data worker process finished due to KeyboardInterrupt in audio_worker")
        finally:
            # After recording stops, feed remaining buffer data
            if buffer:
                audio_queue.put(bytes(buffer))
            try:
                if audio_input:
                    audio_input.cleanup()
            except Exception as e:
                print(f"Exception encountered during audio_input cleanup in audio_worker {e}")

    def wakeup(self):
        """Wakeup the audio_input"""
        self.listen_start = time.time()

    def start(self):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        # Ensure there's a minimum interval
        # between stopping and starting recording
        #if (time.time() - self.recording_stop_time
        #        < 0.0):
        if (time.time() - self.recording_stop_time) < 5:
            print("AudioRecorder.start: Attempted to start recording too soon after stopping.")
            return self

        if self.debug:
            print("AudioRecorder.start: recording started")
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
            print("AudioRecorder.stop: Attempted to stop recording too soon after starting")
            return self

        if self.debug:
            print("AudioRecorder.stop: recording stopped")
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
        try:
            if self.debug:
                print(f"START wait_audio, if listen_start is 0 then we will set it now. listen_start")
            if self.listen_start == 0:
                self.listen_start = time.time()

            # If not already recording, wait for voice activity to start
            if not self.is_recording and not self.frames:
                self._set_state("listening")
                self.start_recording_on_voice_activity = True

                if self.debug:
                    print("AudioRecorder.wait_audio: Waiting for recording to start...")
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout = 0.04): # DAB: Original value 0.04
                        if self.debug:
                            print("AudioRecorder.wait_audio: BREAK on start_recording_event")
                        break

            # if recording is ongoing, then wait for voice inactivity
            if self.is_recording:
                self.stop_recording_on_voice_deactivity = True
                if self.debug:
                    print("AudioRecorder.wait_audio: Waiting for recording to stop...")
                while not self.interrupt_stop_event.is_set():
                    if (self.stop_recording_event.wait(timeout = 0.02)): # DAB: Originally 0.02
                        if self.debug:
                            print("AudioRecorder.wait_audio: BREAK on stop_recording_event")
                        break
            frames = self.frames
            self.is_silero_speech_active = False
            self.is_webrtc_speech_active = False
            self.silero_check_time = 0

            audio_array = np.frombuffer(b"".join(frames), dtype=np.int16)
            self.audio = audio_array
            #self.audio = audio_array.astype(np.float32) # DAB: This is probably not necessary.
            self.frames.clear()
            self.recording_stop_time = 0
            self.listen_start = 0
            self._set_state("inactive")
            return self.audio
        except KeyboardInterrupt:
            self.shutdown()
            raise

    def wait_rev_audio(self):
        """Waits for the start and completion of the recording process"""
        try:
            if self.debug:
                print(f"START wait_audio, if listen_start is 0 then we will set it now. listen_start is {self.listen_start}")
            if self.listen_start == 0:
                self.listen_start = time.time()

            # If not already recording, wait for voice activity to start
            if not self.is_recording and not self.frames:
                self._set_state("listening")
                self.start_recording_on_voice_activity = True

                if self.debug:
                    print("AudioRecorder.wait_audio: Waiting for recording to start...")
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout = 0.04): # DAB: Original value 0.04
                        if self.debug:
                            print("AudioRecorder.wait_audio: BREAK on start_recording_event")
                        break
            # if recording is ongoing, then wait for voice inactivity
            if self.is_recording:
                self.stop_recording_on_voice_deactivity = True
                if self.debug:
                    print("AudioRecorder.wait_audio: Waiting for recording to stop...")
                while not self.interrupt_stop_event.is_set():
                    if (self.stop_recording_event.wait(timeout = 0.02)): # DAB: Originally 0.02
                        if self.debug:
                            print("AudioRecorder.wait_audio: BREAK on stop_recording_event")
                        break

            if self.debug:
                print("AudioRecorder.wait_audio: About to convert recorded frames to the appropriate format.")
            # Convert recorded frames to the appropriate audio format
            audio_array = np.frombuffer(b"".join(self.frames), dtype=np.int16)
            ret_audio = copy.deepcopy(audio_array.astype(np.float32) / 32768.0)
            self.frames.clear()
            self.clear_audio_queue()
            self.audio = None
            # Reset recording related timestamps
            if self.debug:
                print("AudioRecorder.wait_audio: About to reset recording related timestamps")
            self.is_recording = False
            self.start_recording_on_voice_activity = False
            self.stop_recording_on_voice_deactivity = False

            #self.abort()
            self.recording_stop_time = 0
            self.listen_start = 0
            self._set_state("inactive")

            if self.debug:
                print("END wait_audio")
            return ret_audio
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
            self.shutdown_event.set()
            print("AudioRecorder.shutdown: About to join to recording_thread")
            #if self.recording_thread:
            #    self.recording_thread.join(timeout=1)
            #    if self.recording_thread.is_alive():
            #        print("AudioRecorder.shutdown: Recording thread is still alive...")

            #if self.reader_process:
            #    self.reader_process.join(timeout=1)
            #    if self.reader_process.is_alive():
            #        print("AudioRecorder.shutdown: Reader process is still alive...")
            
            import gc
            gc.collect()

    def _process_wakeword(self, data : bytearray):
        """
        Processes audio data to detect wake words.
        """
        if self.debug:
            print(f"AudioRecorder._process_wakeword: START, data is type {type(data)} and len is {len(data)}")
        pcm = np.frombuffer(data, dtype=np.int16)
        if self.debug:
            print(f"pcm = {type(pcm)} and len = {len(pcm)}")
        prediction = self.openwakeword.predict(pcm)
        if self.debug:
            print(f"prediction: {prediction}")
        max_score = -1
        max_index = -1
        wakewords_in_prediction = len(self.openwakeword.prediction_buffer.keys())
        if self.debug:
            print(f"wakewords in prediction = {wakewords_in_prediction}")
        self.wakewords_sensitivities
        if wakewords_in_prediction:
            for idx, mdl in enumerate(self.openwakeword.prediction_buffer.keys()):
                scores = list(self.openwakeword.prediction_buffer[mdl])
                #if scores[-1] > 0.01:
                if self.debug:
                    print(f"Score = {scores[-1]}")
                    # Test code
                    #testb = pcm.tobytes()
                    #import wave
                    #with wave.open("/tmp/maybetest.wav", "wb") as wv:
                    #    wv.setnchannels(1)
                    #    wv.setframerate(16000)
                    #    wv.setsampwidth(2)
                    #    wv.writeframes(testb)
                if scores[-1] >= self.wakewords_sensitivity and scores[-1] > max_score:
                    max_score = scores[-1]
                    max_index = idx
            return max_index  
        else:
            return -1

    def get_audio_for_transcription(self):
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        aud = None
        try:
            aud = self.wait_audio()
        except KeyboardInterrupt:
            print("AudioRecorder.get_audio_for_transcription: KeyboardInterrupt in text() method")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

        if self.is_shutdown or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return None

        return aud

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

            print("Initialized recording worker")
            while self.is_running:
                time.sleep(0.01)
                if self.debug:
                    print("START _recording_worker iteration!")
                if last_inner_try_time:
                    last_processing_time = time.time() - last_inner_try_time
                    if self.debug:
                        print(f"AudioRecorder._recording_worker: last_processing_time = {last_processing_time}")
                    if last_processing_time > 0.1: # DAB: Original value 0.1
                        pass
                        #print("recording_worker: Processing took a bit too long...")
                last_inner_try_time = time.time()
                try:
                    try:
                        data = self.audio_queue.get(timeout=0.01) # DAB: Original value 0.01
                        #self.debug_audio(data)
                        if self.debug:
                            print(f"DAB: Found some data in audio_queue: {type(data)}")
                    except queue.Empty:
                        if not self.is_running:
                            print("recording_worker: Not running, breaking loop")
                            break
                        continue
                    # Is there a callback defined for on_recorded_chunk?
                    if self.on_recorded_chunk:
                        self.on_recorded_chunk(data)

                    # Check handle buffer overflow, assume True for now
                    #if self.audio_queue.qsize() > 50:  # allowed latency limit
                    #    print("recording_worker: queue size exceeds limit")

                    while self.audio_queue.qsize() > 50:
                        data = self.audio_queue.get()
                except BrokenPipeError:
                    print("recording_worker: broken pipe error")
                    self.is_running = False
                    break

                if time_since_last_buffer_message:
                    time_passed = time.time() - time_since_last_buffer_message
                    #print(f"AudioRecorder._recording_worker: time_passed since last buffer message = {time_passed}")
                    if time_passed > 2: # DAB: Original value is 1
                        #print("AudioRecorder._recording_worker: time_passed is more than 1 so reseting!!!!!!!")
                        time_since_last_buffer_message = time.time()
                else:
                    time_since_last_buffer_message = time.time()

                failed_stop_attempt = False

                #print("AudioRecorder._recording_worker: Check if self.is_recording. If not recording then we can look for wakewords")
                if not self.is_recording:
                    #print("AudioRecorder._recording_worker: not recording...")
                    time_since_listen_start = (time.time() - self.listen_start if self.listen_start else 0)
                    #print(f"AudioRecorder._recording_worker: time_since_listen_start = {time_since_listen_start}")
                    wakeword_activation_delay_passed = (
                        time_since_listen_start > self.wakeword_activation_delay
                    )
                    #print(f"AudioRecorder._recording_worker: time_since_listen_start[{time_since_listen_start}] > self.wakeword_activation_delay[{self.wakeword_activation_delay}]")
                    #if wakeword_activation_delay_passed:
                    #    print(f"AudioRecorder._recording_worker: wakeword_activation_delay_passed {wakeword_activation_delay_passed} and delay_was_passed {delay_was_passed}")
                    if wakeword_activation_delay_passed and not delay_was_passed:
                        #print(f"AudioRecorder._recording_worker: self.wakeword_activation_delay {self.wakeword_activation_delay}")
                        if self.wakeword_activation_delay:
                            if self.on_wakeword_timeout:
                                self.on_wakeword_timeout()
                    delay_was_passed = wakeword_activation_delay_passed
                    
                    # Set state
                    if not self.recording_stop_time:
                        #print(f"AudioRecorder._recording_worker: We are inside the if not self.recording_stop_time block! {wakeword_activation_delay_passed} and delay_passed: {delay_was_passed}")
                        if wakeword_activation_delay_passed and not self.wakeword_detected:
                            self._set_state("wakeword")
                        else:
                            if self.listen_start:
                                self._set_state("listening")
                            else:
                                self._set_state("inactive")
                    if wakeword_activation_delay_passed:
                        #print("AudioRecorder._recording_worker: wakeword_activation_delay_passed is True")
                        try:
                            #TODO: This is supposed to return an index where 0 means no wakeword was found, and more
                            # than 0 means that it was found. I think it expects this number to represent a number of 
                            # word samples that should be removed from the audio buffer?
                            wakeword_index = self._process_wakeword(data)
                            #print(f"AudioRecorder._recording_worker: wakeword_index is {wakeword_index}")
                        except struct.error:
                            print("Recording Worker: Error unpacking audio data")
                            continue
                        except Exception as e:
                            print(f"Recording Worker: Wakeword processing error {e}")
                            continue

                        # if a wakeword is detected
                        if wakeword_index >= 0:
                            #print(f"AudioRecorder._recording_worker: The wakeword_index is >= 0 therefore wakeword is detected!!! YAY!")
                            wakeword_samples_to_remove = int(self.sample_rate * self.wakeword_buffer_duration)
                            self.wakeword_detected = True
                            if self.on_wakeword_detected:
                                self.on_wakeword_detected()

                    # Check for voice activity to trigger start of recording
                    if (not wakeword_activation_delay_passed and self.start_recording_on_voice_activity) or self.wakeword_detected:
                        #print("AudioRecorder._recording_worker: about to check if voice is active..")
                        if self._is_voice_active():
                            #print("AudioRecorder._recording_worker: _is_voice_active returned True so about to start the AudioRecorder")
                            self.start()
                            self.start_recording_on_voice_activity = False
                            # Add the buffered audio
                            # to the recording frames
                            #print(f"AudioRecorder._recording_worker: Just a quick frames length check {len(self.frames)}")
                            self.frames.extend(list(self.audio_buffer))
                            self.audio_buffer.clear()
                            self.silero_vad_model.reset_states()
                        else:
                            #print("AudioRecorder._recording_worker: _is_voice_active is FALSE, so checking a copy of data to _check_voice_activity")
                            data_copy = data[:]
                            self._check_voice_activity(data_copy)
                    #print("AudioRecorder._recording_worker: the last thing we do at the end of not recording is set self.speech_end_silence_start to 0")
                    self.speech_end_silence_start = 0
                else:
                    # If we are currently recording
                    #print("AudioRecorder._recording_worker: We are actually currently recording!")
                    if wakeword_samples_to_remove and wakeword_samples_to_remove > 0:
                        #print(f"AudioRecorder._recording_worker: There are wakeword samples to remove... wakeword_samples_to_remove = {wakeword_samples_to_remove} and wakeword_detected = {self.wakeword_detected} PURPLEMONKEYDISHWASHER")
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
                    #print(f"AudioRecoder._recording_worker: about to check stop_recording_on_voice_deactivity = {self.stop_recording_on_voice_deactivity}") 
                    if self.stop_recording_on_voice_deactivity:
                        #print(f"Stopping the recording after detection of silence...")
                        is_speech = self._is_silero_speech(data)
                        #print(f"AudioRecorder._recording_worker: is_speech = {is_speech}")
                        
                        if not is_speech:
                            # Voice deactivity was detected, so we start
                            # measuring silence time before stopping recording
                            if self.speech_end_silence_start == 0 and \
                                (time.time() - self.recording_start_time > self.min_length_of_recording):
                                #print("AudioRecorder._recording_worker: So we did detect voice deactivity, speech_end_silence_start is 0 and we reached the min_length_of_recording parameter. Now set speech_end_silence_start to start now()")
                                self.speech_end_silence_start = time.time()
                        else:
                            if self.speech_end_silence_start:
                                self.speech_end_silence_start = 0
                        # Wait for silence to stop recording after speech
                        if self.speech_end_silence_start and time.time() - \
                                self.speech_end_silence_start >= \
                                self.post_speech_silence_duration:
                            #print("AudioRecorder._recording_worker: The silence started, and we exceeded the post_speech_silence_duration")
                            # Calculate time difference
                            time_diff = time.time() - self.speech_end_silence_start
                            #print(f"time_diff from speech_end_silence_start: {time_diff}")

                            #print("AudioRecorder._recording_worker: Finally, we can append the data to self.frames and invoke self.stop()!")
                            self.frames.append(data)
                            self.stop()
                            #print(f"AudioRecorder._recording_worker: Stop() has been invoked and we are about to perform a final check to make sure that we are no longer recording. If the stop attempt was successful then we will set self.speech_end_silence_start to 0")
                            if not self.is_recording:
                                self.speech_end_silence_start = 0
                            else:
                                print("Failed stop attempt is true!")
                                failed_stop_attempt = True

                if not self.is_recording and was_recording:
                    # Reset after stopping recording to ensure clean state
                    self.stop_recording_on_voice_deactivity = False
                    #print("AudioRecorder._recording_worker: self.is_recording is False and was_recording is True, therefore we are setting self.stop_recording_on_voice_deactivity to False.")
                #else:
                #    print(f"AudioRecorder._recording_worker: self.is_recording is {self.is_recording} and was_recording is {was_recording}, therefore we are NOT setting self.stop_recording_on_voice_deactivity to False.  Not sure what this means yet though...")
                
                tem = (time.time() - self.silero_check_time)
                #print(f"AudioRecorder._recording_worker: The silero_check_time duration is {tem}, if it exceeds 0.1 then self.silero_check_time will be set to 0.")
                #if tem > 0.1:
                #    self.silero_check_time = 0

                # Handle wake word timeout (waited to long initiating
                # speech after wake word detection)
                tem = time.time() - self.wakeword_detect_time
                #print(f"AudioRecorder._recording_worker: The wakeword detect time duration is {tem} which if it is higher than {self.wakeword_timeout} then self.wakeword_detect_time will be set to 0 and self.wakeword_detected will be set to False")
                if self.wakeword_detect_time and time.time() - \
                        self.wakeword_detect_time > self.wakeword_timeout:

                    self.wakeword_detect_time = 0
                    if self.wakeword_detected and self.on_wakeword_timeout:
                        self.on_wakeword_timeout()
                    self.wakeword_detected = False

                #print(f"AudioRecorder._recording_worker: was_recording will be set to self.is_recording which is currently {self.is_recording}")
                was_recording = self.is_recording

                #print(f"AudioRecorder._recording_worker: about to check if we are currently recording and have no failed stop events. If so then data gets appended to self.frames! failed_stop_event by the way is {failed_stop_attempt}")
                if self.is_recording and not failed_stop_attempt:
                    self.frames.append(data)

                if not self.is_recording or self.speech_end_silence_start:
                    #print(f"SET data to self.audio_buffer if not recording or silence started... self.is_recording = {self.is_recording} and self.speech_end_silence_start = {self.speech_end_silence_start}")
                    self.audio_buffer.append(data)

                #print("END _recording_worker iteration!")
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
        return self.is_webrtc_speech_active

    def _is_silero_speech(self, chunk):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        from scipy import signal
        #print("DAB: Check for speech in is_silero_speech...")
        timing_start = time.time()

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
        #print(f"{rd_prob}")
        is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
        #if is_silero_speech_active:
        #    print(f"Silero detected speech activity...")
        self.is_silero_speech_active = is_silero_speech_active
        self.silero_working = False
        #print(f"DAB: is_silero_speech_active = {is_silero_speech_active}")
        #print(f"AudioRecorder.wait_audio: silero speech time = {time.time() - timing_start}")
        return is_silero_speech_active

    def _check_voice_activity(self, data):
        """
        Initiate check if voice is active based on the provided data.

        Args:
            data: The audio data to be checked for voice activity.
        """
        return self._is_webrtc_speech(data)
        #print("DAB: Checking voice activity...")
        #if not self.silero_working:
        #    self.silero_working = True

            # Run the intensive check in a separate thread
            #threading.Thread(
            #    target=self._is_silero_speech,
            #    args=(data,)).start()

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
        return self.is_webrtc_speech_active

    def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        # Number of audio frames per millisecond
        frame_length = int(16000 * 0.01)  # for 10ms frame
        num_frames = int(len(chunk) / (2 * frame_length))
        speech_frames = 0

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = chunk[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames_must_be_true:
                    #print(f"Speech detected in frame {i + 1} of {num_frames}")
                    self.is_webrtc_speech_active = True
                    return True
        if all_frames_must_be_true:
            speech_detected = speech_frames == num_frames
            self.is_webrtc_speech_active = speech_detected
            return speech_detected
        else:
            #print(f"Speech not detected in any of {num_frames} frames")
            self.is_webrtc_speech_active = False
            return False
    
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
