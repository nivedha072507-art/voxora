import pyaudio

p = pyaudio.PyAudio()

DEVICE = 8
info = p.get_device_info_by_index(DEVICE)

print("Device:", info["name"])
print("Sample rate:", int(info["defaultSampleRate"]))
print("Input channels:", info["maxInputChannels"])

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    input_device_index=DEVICE,
    frames_per_buffer=1024
)

print("Microphone stream opened.")
print("Speak for 5 seconds...")

for i in range(0, int(16000 / 1024 * 5)):
    data = stream.read(1024, exception_on_overflow=False)

print("Audio captured successfully.")

stream.stop_stream()
stream.close()
p.terminate()

print("Microphone test PASSED.")
