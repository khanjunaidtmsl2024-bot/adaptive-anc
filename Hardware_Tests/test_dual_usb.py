import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time

# ============================================================
# USB DUAL-MIC SYNCHRONIZATION TEST
# ============================================================

FS = 48000
CHANNELS = 1
DURATION = 30
BLOCKSIZE = 480

# Windows WASAPI device numbers
USB1 = 17
USB2 = 18

recorded_usb1 = []
recorded_usb2 = []


def callback_usb1(indata, frames, time_info, status):
    if status:
        print("USB #1:", status)

    recorded_usb1.append(indata.copy())


def callback_usb2(indata, frames, time_info, status):
    if status:
        print("USB #2:", status)

    recorded_usb2.append(indata.copy())


print("==============================================")
print("       DUAL USB MICROPHONE TEST")
print("==============================================")
print()
print("USB #1 : device", USB1)
print("USB #2 : device", USB2)
print("Sample rate :", FS, "Hz")
print("Bit depth   : 16-bit")
print("Channels    : 1 (mono)")
print("Duration    :", DURATION, "seconds")
print()

print("Opening USB #1...")

stream_usb1 = sd.InputStream(
    device=USB1,
    samplerate=FS,
    channels=CHANNELS,
    dtype="int16",
    blocksize=BLOCKSIZE,
    callback=callback_usb1
)

print("Opening USB #2...")

stream_usb2 = sd.InputStream(
    device=USB2,
    samplerate=FS,
    channels=CHANNELS,
    dtype="int16",
    blocksize=BLOCKSIZE,
    callback=callback_usb2
)

print()
print("Starting both streams...")
print()

stream_usb1.start()
stream_usb2.start()

print("==============================================")
print("             RECORDING STARTED")
print("==============================================")
print()
print("0-10 seconds  : stay quiet")
print("10-20 seconds : speak normally")
print("20-30 seconds : clap or tap several times")
print()
print("DO NOT MOVE THE MICROPHONES.")
print()

time.sleep(DURATION)

print()
print("Stopping streams...")

stream_usb1.stop()
stream_usb2.stop()

stream_usb1.close()
stream_usb2.close()

print("Streams closed.")
print()

# ============================================================
# COMBINE RECORDED BLOCKS
# ============================================================

audio_usb1 = np.concatenate(recorded_usb1, axis=0).flatten()
audio_usb2 = np.concatenate(recorded_usb2, axis=0).flatten()

# Keep exactly the same number of samples
n = min(len(audio_usb1), len(audio_usb2))

audio_usb1 = audio_usb1[:n]
audio_usb2 = audio_usb2[:n]

# ============================================================
# SAVE WAV FILES
# ============================================================

write(
    "usb1_48k.wav",
    FS,
    audio_usb1
)

write(
    "usb2_48k.wav",
    FS,
    audio_usb2
)

# ============================================================
# BASIC INFORMATION
# ============================================================

print("==============================================")
print("          RECORDING COMPLETE")
print("==============================================")
print()

print("USB #1 samples :", len(audio_usb1))
print("USB #2 samples :", len(audio_usb2))

print(
    "Recorded duration :",
    round(len(audio_usb1) / FS, 3),
    "seconds"
)

print()
print("Files created:")
print("  usb1_48k.wav")
print("  usb2_48k.wav")
print()

print("Next step:")
print("We will analyze these files for:")
print("  1. Cross-correlation")
print("  2. Relative sample delay")
print("  3. Gain difference")
print("  4. Clock drift")
print("  5. Frequency response")
print()