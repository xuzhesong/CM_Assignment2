import wave
import numpy as np
import matplotlib.pyplot as plt

window_length = 5
num_points = 10
def median_filter(audio, detection, size):
    l = len(audio)
    origin = audio.copy() * detection
    n = size // 2
    if(size == 1):
        return origin
    for i in range(0, l - 1):
        if(detection[i] == 0):
            array = origin[i - n : i + n]
            origin[i] = Median(array)
    return origin

def Median(array):
    '''
    Input an array and return the median number of the array
    Args:
        array: an array with n numbers
    Returns:
        array[m_index]: the median nubmer
            m_index: the index of the median number in array
    '''
    n = len(array) 
    m_index = n // 2
    array = sorted(array)
    return array[m_index]





def compute_MSE(clean, restored, detection):
    mse = 0
    n = len(clean)
    clean = clean.copy() / 2
    for  i in range(0, n - 1):
        if(detection[i] == 0):
            mse = mse + abs(clean[i] - restored[i])**2
    return mse/(len(detection)-sum(detection))
    
def test(clean, degraded, detection, num_points):
    mse = [0] * num_points
    win_size = [0] * num_points
    restored = median_filter(degraded, detection, window_length)
    for i in range(0, num_points):
        win_size[i] = i * 2 + 1
        restored = median_filter(degraded, detection, win_size[i])
        mse[i] = compute_MSE(clean, restored, detection)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 1, 1)
    plt.plot(win_size, mse)
    plt.title('MSE')
    plt.xlim(min(win_size), max(win_size))
    plt.xticks(win_size)
    plt.xlabel('X Axis Label')
    plt.ylabel('Y Axis Label')

    plt.show()






'''read audio from .wav files'''
with wave.open('degraded.wav','rb') as degraded_audio:

    frame_rate = degraded_audio.getframerate()
    n_frames = degraded_audio.getnframes()

    degraded = degraded_audio.readframes(n_frames)
    degraded = np.frombuffer(degraded, dtype=np.int16)
    degraded = degraded / 2**15

with wave.open('clean.wav','rb') as clean_audio:

    n_frames = clean_audio.getnframes()
    
    clean = clean_audio.readframes(n_frames)
    clean = np.frombuffer(clean, dtype=np.int16)
    clean = clean / 2**15

with wave.open('detection.wav','rb') as detection_file:
    
    n_frames = detection_file.getnframes()
    
    detection = detection_file.readframes(n_frames)
    detection = np.frombuffer(detection, dtype=np.int16)
    detection = detection / 2**15


# restored = median_filter(degraded, detection, window_length)
# mse = compute_MSE(clean, restored, detection)
# print('mse = ', mse)

test(clean, degraded, detection, num_points)

# plt.figure(figsize=(10, 5))

# plt.subplot(3, 1, 1)
# plt.plot(degraded)
# plt.title('degraded')
# plt.ylim(-1, 1)
# plt.xlabel('X Axis Label')
# plt.ylabel('Y Axis Label')


# plt.subplot(3, 1, 2)
# plt.plot(clean)
# plt.title('clean')
# plt.ylim(-1, 1)
# plt.xlabel('X Axis Label')
# plt.ylabel('Y Axis Label')


# plt.subplot(3, 1, 3)
# plt.plot(restored)
# plt.title('restored')
# plt.ylim(-1, 1)
# plt.xlabel('X Axis Label')
# plt.ylabel('Y Axis Label')

# plt.show()
