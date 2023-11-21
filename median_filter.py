import numpy as np
def median_filter(audio, detection, num):
    l = len(audio)
    n = num // 2
    for i in range(0, l):
        if(detection[i]):
            array = audio[i - n : i + n]
            audio[i] = Median(array)
    return audio

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
