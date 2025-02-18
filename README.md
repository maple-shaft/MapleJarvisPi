# MapleJarvisPi

This project provides a multi-threaded software client that is intended for execution on a Raspberry Pi or other similar resource constrained device. This device will provide the interaction point for a custom MapleJarvis AI assistant, and will establish a connection to a server process in the sibling project, `MapleJarvis`.

### How it works...

The client software performs Voice Activity Detection (VAD) using WebRTC, as well as wake word detection using OpenWakeWord locally off of an audio data queue populated from a real-time audio input source.  Recorded audio is sent to the server process where it is transcribed into text, fed into an LLM to retrieve a text response, synthesizing the model response into audio, and then sent back to the client over a socket. Received audio data is queued at the client and then played to the default audio output device connected to the client host.

### Tested?

It is currently tested on a Raspberry Pi Zero Model 2W (1GHz, 4 cores, armv7l 32 bit architecture, 512MB RAM) and its BARELY functional. If you do want to use this software, I would recommend a less resource constrained host machine.  Python after all is not none for being highly performant, nor efficient.

### Other Dependencies...

There are a handful of required Python libraries that have native C/C++ requirements, and as of early 2025 there existed no trustworthy or up-to-date prebuilt wheels for arm7vl 32 bit CPU architectures on a number of libraries.  Namely, `torch`, `torchaudio`, and `onnxruntime`, which are all required by OpenWakeWord. I had to compile these wheels from source and install them manually, however I am not publicly disclosing the URLs to these artifacts unless I am asked for them.  Feel free to DM me if you would like me to send you a link to download these.