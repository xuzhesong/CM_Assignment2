import wave
import numpy as np
import matplotlib.pyplot as plt

'''read audio from .wav files'''

with wave.open('degraded.wav','rb') as degraded_audio:

    frame_rate = degraded_audio.getframerate()
    n_frames = degraded_audio.getnframes()
    sample_width = degraded_audio.getsampwidth()

    degraded = degraded_audio.readframes(n_frames)

with wave.open('clean.wav','rb') as clean_audio:

    frame_rate = clean_audio.getframerate()
    n_frames = clean_audio.getnframes()
    sample_width = clean_audio.getsampwidth()

    clean = clean_audio.readframes(n_frames)


clean = np.frombuffer(clean, dtype=np.int16)
degraded = np.frombuffer(degraded, dtype=np.int16)

print(clean.shape)
print(degraded.shape)

clean = clean / 2**15
degraded = degraded / 2**15
plt.figure(figsize=(10, 4))

plt.subplot(2, 1, 1)
plt.plot(clean)
plt.title('clean')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')

plt.subplot(2, 1, 2)
plt.plot(degraded)
plt.title('degraded')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')

plt.show()

