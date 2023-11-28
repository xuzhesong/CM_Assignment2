import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.io import wavfile
from math import ceil
import unittest
from tqdm import tqdm
window_length_median = 3
window_length_spline = 11
num_points = 100


##########################################################################

# define some functions
# include medain, spline, median filter, spline filter, MSE, test
#
# medain(): return the medain value of the array
# spline(): return the interpolation value using cubic spline method
# median filter(): processing the entire audio using medain method
# spline filter(): processing the entire audio using cubic spline method
# MSE(): compute MSE between the interpolated signal and clean audio (only for clicks)
# test(): test performance for median filter and spline filter by
# calculate mse using different window length

##########################################################################

def Median(array):
    ''' Median function
    Args:
        array: an array with n numbers
    Returns:
        median_value: the median nubmer
    '''
    n = len(array)
    median_truth = np.median(array)
    m_index = n // 2
    array = sorted(array)
    median_value = array[m_index]

    return median_value


def Spline(array, window_length):
    '''fucntion of cubic spline method
    Args:
        array: an array with n points(n = window_length - 1)
        window length: window length of the filter
    Returns:
        interpolation: the interpolation calculated by spline method
    '''
    l = len(degraded)
    x = list(range(0, window_length - 1))
    y = array.copy()
    cs = CubicSpline(x, y)

    x_fine = np.linspace(0, window_length - 2, window_length)
    y_fine = cs(x_fine)
    interpolation = y_fine[(window_length // 2)]
    return interpolation


def median_filter(audio, detection, window_length):
    ''' function of median filter
    Args:
        audio: degraded audio
        detection: detection array, shows the position of clicks
        window length: window length of the filter

    Return:
        restored: the restored signal after median interpolation

    '''
    l = len(audio)
    restored = audio.copy()
    n = window_length // 2
    if (window_length == 1):
        return restored
    for i in tqdm(range(0, l - 1)):
        if (detection[i] == 0):
            array = restored[i - n: i + n + 1]
            restored[i] = Median(array.copy())
    return restored


def cubic_spline_filter(degraded, detection, window_length):
    '''function of cubic spline filter
    Args:
        audio: degraded audio
        detection: detection array, shows the position of clicks
        window length: window length of the filter

    Return:
        y: the restored signal
    '''
    l = len(degraded)
    y = degraded.copy()
    n = window_length // 2
    for i in tqdm(range(0, l)):
        if (detection[i] == 0):
            array = np.concatenate((y[(i - n): i], y[(i + 1): i + n + 1]))
            y[i] = Spline(array, window_length)
    return y


def compute_MSE(clean, restored, detection):
    '''function of cubic spline filter
    Args:
        clean: clean audio
        restored: restored signal
        detection: detection array, shows the position of clicks

    Return:
        mse: mean square error between clean and restored audio
    '''
    mse = 0
    clicks = 0
    n = len(clean)
    # clean = clean.copy() / 2
    # restored = restored.copy() * 2
    for i in range(0, n - 1):
        if (detection[i] == 0):
            clicks = clicks + 1
            mse = mse + (clean[i] - restored[i])**2
    mse = mse / clicks
    return mse


def test(clean, degraded, detection, num_points):
    ''' test function
    Args:
        clean: clean audio
        degraded: degraded audio
        detection: detection array, shows the position of clicks
        num_points: how many odd points used in test
    Plot:
        mse curve for median method and spline method
    No return
    '''
    mse_median = [0] * num_points
    mse_spline = [0] * num_points
    win_size = [0] * num_points
    low_win = 0
    low_mse = 1
    for i in tqdm(range(1, num_points)):
        win_size[i] = i * 2 + 1
        restored_median = median_filter(degraded, detection, win_size[i])
        restored_spline = cubic_spline_filter(degraded, detection, win_size[i])
        mse_median[i] = compute_MSE(clean, restored_median, detection)
        mse_spline[i] = compute_MSE(clean, restored_spline, detection)
        if mse_spline[i] < low_mse:
            low_win = win_size[i]
            low_mse = mse_spline[i]
    print('low_win = ', low_win)
    print('low_mse = ', low_mse)
    # plot the MSE in one figure
    plt.figure(figsize=(10, 5))

    plt.plot(win_size[1:], mse_median[1:], label='median')
    plt.legend(title='Median filter', frameon=False, ncol=2)
    plt.plot(win_size[1:], mse_spline[1:], label='spline')
    plt.legend(title='cubic spline filter', frameon=False, ncol=2)
    plt.title('MSE')
    plt.xlim(min(win_size), max(win_size))
    plt.xticks(win_size)
    plt.xlabel('X Axis Label')
    plt.ylabel('Y Axis Label')

    plt.show()


##########################################################################
'''read audio from .wav files'''
with wave.open('degraded.wav', 'rb') as degraded_audio:

    frame_rate = degraded_audio.getframerate()
    n_frames = degraded_audio.getnframes()

    degraded = degraded_audio.readframes(n_frames)
    degraded = np.frombuffer(degraded, dtype=np.int16)
    degraded = degraded / 2**15

with wave.open('clean.wav', 'rb') as clean_audio:

    n_frames = clean_audio.getnframes()

    clean = clean_audio.readframes(n_frames)
    clean = np.frombuffer(clean, dtype=np.int16)
    clean = clean / 2**16

with wave.open('detection.wav', 'rb') as detection_file:

    n_frames = detection_file.getnframes()

    detection = detection_file.readframes(n_frames)
    detection = np.frombuffer(detection, dtype=np.int16)
    detection = detection / 2**15

# restored by median method
restored = median_filter(degraded, detection, window_length_median)
# restored by cubic spline method
#restored = cubic_spline_filter(degraded, detection, window_length_spline)

# compute the MSE for the restored signal
mse = compute_MSE(clean, restored, detection)
print('mse = ', mse)

# run test function
#test(clean, degraded, detection, num_points)

# restored = restored * 2 ** 16
# restored = restored.astype(np.int16)
# wavfile.write('output_cubicSplines.wav', frame_rate, restored)


# plot the degraded, clean, restored signal
plt.figure(figsize=(10, 5))

plt.subplot(3, 1, 1)
plt.plot(degraded)
plt.title('degraded')
plt.ylim(-1, 1)


plt.subplot(3, 1, 2)
plt.plot(clean)
plt.title('clean')
plt.ylim(-1, 1)


plt.subplot(3, 1, 3)
plt.plot(restored)
plt.title('restored')
plt.ylim(-1, 1)

plt.show()

##########################################################################
# unittest


class TestMethods(unittest.TestCase):
    ''' unittest '''

    def test_median(self):
        for i in range(0, len(degraded) - 1):
            n = window_length_median // 2
            if (detection[i] == 0):
                array = restored[i - n: i + n + 1].copy()
                M = np.median(array)
                m = Median(array)
                self.assertEqual(m, M)


if __name__ == '__main__':
    unittest.main()
##########################################################################
