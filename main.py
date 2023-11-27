import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from math import ceil

window_length = 101
num_points = 50

def median_filter(audio, detection, size):
    l = len(audio)
    origin = audio.copy()
    n = size // 2
    if(size == 1):
        return origin
    for i in range(0, l - 1):
        if(detection[i] == 0):
            array = origin[i - n : i + n + 1]
            origin[i] = Median(array.copy())
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

def Spline(array, size):
    l = len(degraded)
    x = list(range(0, size - 1))
    y = array.copy()
    cs = CubicSpline(x, y)
    
    x_fine = np.linspace(0, size-2, size)
    y_fine = cs(x_fine)
    return y_fine[(size//2)]


def cubic_spline_filter(degraded, detection, size):
    l = len(degraded)
    y = degraded.copy()
    n = size // 2
    for i in range(0, l):
        if(detection[i] == 0):
            array = np.concatenate((y[(i - n) : i], y[(i + 1): i + n + 1]))
            y[i] = Spline(array, size)
    return y

def compute_MSE(clean, restored, detection):
    mse = 0
    clicks = 0
    n = len(clean)
    #clean = clean.copy() / 2
    #restored = restored.copy() * 2
    for  i in range(0, n - 1):
        if(detection[i] == 0):
            clicks = clicks + 1
            mse = mse + (clean[i] - restored[i])**2
    return mse/clicks
    
def test(clean, degraded, detection, num_points):
    mse_median = [0] * num_points
    mse_spline = [0] * num_points
    win_size = [0] * num_points
    for i in range(1, num_points):
        win_size[i] = i * 2 + 1
        restored_median = median_filter(degraded, detection, win_size[i])
        restored_spline = cubic_spline_filter(degraded, detection, win_size[i])
        mse_median[i] = compute_MSE(clean, restored_median, detection)
        mse_spline [i] = compute_MSE(clean, restored_spline , detection)
    
    plt.figure(figsize=(10, 5))

    plt.plot(win_size[1:], mse_median[1:], label = 'median')
    plt.legend(title = 'Median filter', frameon=False, ncol=2)
    plt.plot(win_size[1:], mse_spline[1:], label = 'spline')
    plt.legend(title = 'cubic spline filter', frameon=False, ncol=2)
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
    clean = clean / 2**16

with wave.open('detection.wav','rb') as detection_file:
    
    n_frames = detection_file.getnframes()
    
    detection = detection_file.readframes(n_frames)
    detection = np.frombuffer(detection, dtype=np.int16)
    detection = detection / 2**15


restored = cubic_spline_filter(degraded, detection, window_length)
mse = compute_MSE(clean, restored, detection)

print('mse = ', mse)

test(clean, degraded, detection, num_points)

plt.figure(figsize=(10, 5))

plt.subplot(3, 1, 1)
plt.plot(degraded)
plt.title('degraded')
plt.ylim(-1, 1)
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')


plt.subplot(3, 1, 2)
plt.plot(clean)
plt.title('clean')
plt.ylim(-1, 1)
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')


plt.subplot(3, 1, 3)
plt.plot(restored)
plt.title('restored')
plt.ylim(-1, 1)
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')

plt.show()
